import nest_asyncio
import collections
import functools
import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import calculateFactor as factor

nest_asyncio.apply()


"""
准备输入数据
"""
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(cache_dir=r'D:\Develop\.tff')
NUM_CLIENTS = 10
BATCH_SIZE = 20


def preprocess(dataset):
    """
    数据集预处理
    :param dataset: 数据集
    :return: 展平后的数据集
    """

    def batch_format_fn(element):
        """Flatten a batch of EMNIST data and return a (features, label) tuple."""
        return tf.reshape(element['pixels'], [-1, 784]), tf.reshape(element['label'], [-1, 1])

    return dataset.batch(BATCH_SIZE).map(batch_format_fn)


# 选择客户端并对其数据预处理
client_ids = sorted(emnist_train.client_ids)[:NUM_CLIENTS]
tf_datasets = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in client_ids]

tf_train_datasets = tf_datasets
tf_test_datasets = tf_datasets

tf_model_input_shape = tf_train_datasets[0].element_spec[0].shape
tff_model_input_spec = tf_train_datasets[0].element_spec


# build model
class UserEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_latent_factors, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_latent_factors = num_latent_factors

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, self.num_latent_factors),
            initializer='uniform',
            dtype=tf.float32,
            name='UserEmbeddingKernel'
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.embedding

    def compute_output_shape(self):
        return (1, self.num_latent_factors)


def tf_model():
    """
    使用与图像分类教程中相同的模型。这个模型有一个隐藏层，后跟一个softmax层。
    :return: 创建的模型
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(11, kernel_initializer=initializer, name="Dense"),
        tf.keras.layers.Softmax(name="Softmax"),
    ])


model_fn = functools.partial(tf_model)

ModelVariables = collections.namedtuple('ModelVariables', 'num_examples, loss_sum')
loss_fn = tf.keras.losses.MeanSquaredError()
metrics_fn = lambda: tf.keras.metrics.Mean()


def create_variables():
    return ModelVariables(
        num_examples=tf.Variable(0.0, name="num_examples", trainable=False),
        loss_sum=tf.Variable(0.0, name="loss_sum", trainable=False)
    )


def predict_on_batch(model, x):
    return model(x)


def tf_forward_pass(model, variables, batch):
    pred = model(batch[0])
    loss = loss_fn(batch[1], pred)
    num_examples = tf.cast(tf.size(batch[1]), tf.float32)
    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    return loss, pred


def get_local_metrics(variables):
    return collections.OrderedDict(
        num_examples=variables.num_examples,
        loss=variables.loss_sum / variables.num_examples
    )


@tff.federated_computation
def aggregate_metrics_across_clients(metrics):
    return collections.OrderedDict(
        num_examples=tff.federated_sum(metrics.num_examples),
        loss=tff.federated_mean(metrics.loss, metrics.num_examples)
    )


class tff_model(tff.learning.Model):

    def reset_metrics(self) -> None:
        pass

    def __init__(self) -> None:
        self._variables = create_variables()
        self._model = model_fn()
        self._model.build(input_shape=tf_model_input_shape)

    @property
    def trainable_variables(self):
        return self._model.trainable_variables

    @property
    def non_trainable_variables(self):
        return self._model.non_trainable_variables

    @property
    def local_variables(self):
        return [self._variables.num_examples, self._variables.loss_sum,
                self._variables.accuracy_sum]

    @property
    def local_model_variables(self):
        return self._model.get_layer(name="Dense").weights

    @property
    def global_model_variables(self):
        return self._model.get_layer(name="Softmax").weights

    @property
    def input_spec(self):
        return tff_model_input_spec

    @tf.function
    def predict_on_batch(self, x, training=True):
        del training
        return predict_on_batch(self._model, x)

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = tf_forward_pass(self._model, self._variables, batch)
        num_examples = tf.shape(batch[0])[0]
        return tff.learning.BatchOutput(
            loss=loss,
            predictions=predictions,
            num_examples=num_examples
        )

    @tf.function
    def report_local_outputs(self):
        return get_local_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_metrics_across_clients

    @tf.function
    def report_local_unfinalized_metrics(self):
        return collections.OrderedDict(
            num_examples=[self._variables.num_examples],
            loss=[self._variables.loss_sum, self._variables.num_examples]
        )

    def metric_finalizers(self):
        return collections.OrderedDict(
            num_examples=tf.function(func=lambda x: x[0]),
            loss=tf.function(func=lambda x: x[0] / x[1])
        )


# iterative process
@attr.s(eq=False, frozen=True, slots=True)
class ClientState():
    index = attr.ib()
    local_weights = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput():
    weights_delta = attr.ib()
    client_weight = attr.ib()
    loss = attr.ib()
    client_state = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState():
    model_weights = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
    model_weights = attr.ib()


@tf.function
def server_update(model, server_state, server_optimizer, weights_delta):
    weights = model.global_model_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), weights, server_state.model_weights)

    neg_weights_delta = [-1.0 * x for x in weights_delta]
    server_optimizer.apply_gradients(zip(neg_weights_delta, weights))
    return tff.structure.update_struct(
        server_state,
        model_weights=weights
    )


@tf.function
def build_server_broadcast_message(server_state):
    return BroadcastMessage(
        model_weights=server_state.model_weights)


@tf.function
def client_update(model, dataset, client_state, server_message, client_optimizer, local_optimizer):
    weights_local = model.local_model_variables
    weights_global = model.global_model_variables
    tf.nest.map_structure(
        lambda x, y: x.assign(y),
        weights_local,
        client_state.local_weights
    )
    tf.nest.map_structure(
        lambda x, y: x.assign(y),
        weights_global,
        server_message.model_weights
    )
    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0.0, dtype=tf.float32)

    for batch in dataset:
        with tf.GradientTape(persistent=True) as tape:
            outputs = model.forward_pass(batch)
        grads_local = tape.gradient(outputs.loss, weights_local)
        grads_global = tape.gradient(outputs.loss, weights_global)

        localgrads_and_vars = zip(grads_local, weights_local)
        globalgrads_and_vars = zip(grads_global, weights_global)

        client_optimizer.apply_gradients(globalgrads_and_vars)
        local_optimizer.apply_gradients(localgrads_and_vars)
        batch_size = (tf.shape(batch[0])[0])
        num_examples += batch_size
        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

    weights_delta = tf.nest.map_structure(
        lambda x, y: x - y, weights_global, server_message.model_weights
    )
    client_weights = tf.cast(num_examples, tf.float32)
    return ClientOutput(
        weights_delta,
        client_weights,
        loss_sum / tf.cast(client_weights, dtype=tf.float32),
        ClientState(
            index=client_state.index,
            local_weights=weights_local
        )
    )


def build_federated_averaging_process(
        model_fn, client_state_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
        local_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01)):
    whimsy_model = model_fn()

    @tff.tf_computation
    def server_init():
        model = model_fn()
        return ServerState(
            model_weights=model.global_model_variables)

    server_state_type = server_init.type_signature.result
    model_weights_type = server_state_type.model_weights

    client_state_type = tff.framework.type_from_tensors(client_state_fn())

    @tff.tf_computation(server_state_type, model_weights_type)  # pytype: disable=attribute-error  # gen-stub-imports
    def server_update_fn(server_state, model_delta):
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        return server_update(model, server_state, server_optimizer, model_delta)

    @tff.tf_computation(server_state_type)
    def server_message_fn(server_state):
        return build_server_broadcast_message(server_state)

    server_message_type = server_message_fn.type_signature.result
    tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)

    @tff.tf_computation(tf_dataset_type, client_state_type, server_message_type)
    def client_update_fn(dataset, client_state, server_message):
        model = model_fn()
        client_optimizer = client_optimizer_fn()
        local_optimizer = local_optimizer_fn()
        return client_update(model, dataset, client_state, server_message, client_optimizer, local_optimizer)

    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(tf_dataset_type)
    federated_client_state_type = tff.type_at_clients(client_state_type)

    @tff.federated_computation(federated_server_state_type, federated_dataset_type, federated_client_state_type)
    def run_one_round(server_state, federated_dataset, client_states):
        server_message = tff.federated_map(server_message_fn, server_state)
        server_message_at_client = tff.federated_broadcast(server_message)

        client_outputs = tff.federated_map(client_update_fn,
                                           (federated_dataset, client_states, server_message_at_client))
        weight_denom = client_outputs.client_weight
        round_model_delta = tff.federated_mean(client_outputs.weights_delta, weight=weight_denom)
        round_loss = tff.federated_mean(client_outputs.loss, weight=weight_denom)
        server_state = tff.federated_map(server_update_fn, (server_state, round_model_delta))

        return server_state, round_loss, client_outputs.client_state

    @tff.federated_computation
    def server_init_tff():
        return tff.federated_value(server_init(), tff.SERVER)

    return tff.templates.IterativeProcess(initialize_fn=server_init_tff, next_fn=run_one_round)


def generate_client_state():
    model = tff_model()
    weights = model.local_model_variables
    return ClientState(index=-1, local_weights=weights)


# main
iterative_process = build_federated_averaging_process(tff_model, generate_client_state)

server_state = iterative_process.initialize()

client_states = {i: ClientState(i, generate_client_state().local_weights) for i in range(NUM_CLIENTS)}
for round_num in range(20):
    selected_dataset = tf_test_datasets
    sampled_client_states = [client_states[0] for i in range(NUM_CLIENTS)]
    server_state, trained_loss, updated_client_states = iterative_process.next(
        server_state, selected_dataset, sampled_client_states
    )
    print(f'Round {round_num + 1} training loss: {trained_loss}')
    for client_state in updated_client_states:
        client_index = client_state.index
        tf.nest.map_structure(lambda x, y: x.assign(y),
                              client_states[client_index].local_weights, client_state.local_weights)

clientStates = []
for i in range(NUM_CLIENTS):
    currentClientState = client_states.get(i)
    weight = np.array(currentClientState.local_weights[0]).tolist()
    flatWeight = []
    for value in weight:
        flatWeight.extend(value)
    clientStates.append(factor.State(i, flatWeight))

# TODO 无法获取服务器状态
serverWeight = []
for i in range(11*784):
    serverWeight.append(0)
serverState = factor.State('server', serverWeight)

clf = factor.trainOutlierClassifier(clientStates, serverState)

err = factor.getGroupError(clientStates, serverState)
flag = factor.classifyOutlier(clf, clientStates)
for client in clientStates:
    client.addErr(err)
    client.addFlag(flag)
    client.getCreditScore()

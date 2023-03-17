import tensorflow as tf
import tensorflow_federated as tff

"""
联邦学习算法有4个主要组件：
1.服务器到客户端广播步骤。
2.本地客户端更新步骤。
3.客户端到服务器的上载步骤。
4.服务器更新步骤。
"""

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
        return (tf.reshape(element['pixels'], [-1, 784]), tf.reshape(element['label'], [-1, 1]))

    return dataset.batch(BATCH_SIZE).map(batch_format_fn)


# 选择客户端并对其数据预处理
client_ids = sorted(emnist_train.client_ids)[:NUM_CLIENTS]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in client_ids]


def create_keras_model():
    """
    使用与图像分类教程中相同的模型。这个模型有一个隐藏层，后跟一个softmax层。
    :return: 创建的模型
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(11, kernel_initializer=initializer),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    """
    将 Keras 模型包装为 tff.learning.Model
    :return: 包装后的模型
    """
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)

    return client_weights


@tf.function
def server_update(model, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, mean_client_weights)
    return model_weights


# 创建联合数据：{类型}@放置位置
federated_float_on_clients = tff.FederatedType(tf.float32, tff.CLIENTS)
# str(federated_float_on_clients) -> '{float32}@CLIENTS'


@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def get_average_temperature(client_temperatures):
    return tff.federated_mean(client_temperatures)


@tff.tf_computation
def server_init():
    """
    获取初始化服务器模型参数
    :return: 服务器模型参数
    """
    model = model_fn()
    return model.trainable_variables


@tff.federated_computation
def initialize_fn():
    """
    初始化服务器模型
    :return: 服务器的联合数据
    """
    return tff.federated_value(server_init(), tff.SERVER)


whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
# str(tf_dataset_type) -> '<float32[?,784],int32[?,1]>*' -> 数据集类型签名
model_weights_type = server_init.type_signature.result
# str(model_weights_type) -> '<float32[784,10],float32[10]>' -> 模型的架构


@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    return client_update(model, tf_dataset, server_weights, client_optimizer)


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)


federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    """
    将联邦学习的四部分组合起来
    :param server_weights:服务器训练权重
    :param federated_dataset:联邦学习数据
    :return:服务器更新后的权重
    """
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_weights)

    # Each client computes their updated weights.
    client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

    # The server averages these updates.
    mean_client_weights = tff.federated_mean(client_weights)

    # The server updates its model.
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights


federated_algorithm = tff.templates.IterativeProcess(initialize_fn=initialize_fn, next_fn=next_fn)
# str(federated_algorithm.initialize.type_signature) -> 迭代过程的 and 函数的类型签名
# str(federated_algorithm.next.type_signature) -> 接受服务器模型和客户端数据，并返回更新的服务器模型

# 创建一个集中式评估数据集，然后应用用于训练数据的相同预处理
central_emnist_test = emnist_test.create_tf_dataset_from_all_clients()
central_emnist_test = preprocess(central_emnist_test)


def evaluate(server_state):
    """
    接受服务器状态，并使用 Keras 对测试数据集进行评估
    :param server_state: 服务器状态
    :return: None
    """
    keras_model = create_keras_model()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    keras_model.set_weights(server_state)
    keras_model.evaluate(central_emnist_test)


server_state = federated_algorithm.initialize()

# print(str(server_state[0]))
evaluate(server_state)

for round in range(15):
    server_state = federated_algorithm.next(server_state, federated_train_data)
evaluate(server_state)

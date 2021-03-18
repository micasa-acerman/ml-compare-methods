import skimage.transform
import tensorflow as tf
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist


def get_letter(data, start, end):  # получаем букву из слова, записываем в матрицу
    matrix = []
    row = []
    for i in range(len(data)):
        row = []
        for j in range(start, end):
            row.append(data[i][j])
        matrix.append(row)
    return matrix


def crop_borders(data):  # обрезаем лишние пиксели сверху и снизу
    min_l = 999999
    min_u = 999999
    max_d = 0
    max_r = 0
    for j, row in enumerate(data):
        for i in range(len(row)):
            if row[i] == 1:
                if j < min_u:
                    min_u = j
                if j > max_d:
                    max_d = j
                if i < min_l:
                    min_l = i
                if i > max_r:
                    max_r = i

    data = data[min_u:max_d + 1]
    for i in range(len(data)):
        data[i] = data[i][min_l:max_r + 1]
    return data


def create_vectors(image):  # создаем список, в котором будут храниться буквы в матричном виде

    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    letters = []
    data = []

    for i in range(height):
        row = []
        for j in range(width):
            a = pix[j, i][0]
            b = pix[j, i][1]
            c = pix[j, i][2]
            if a == 255 and b == 255 and c == 255:
                row.append(0)  # white — 0
            else:
                row.append(1)  # black — 1
        data.append(row)

    data = crop_borders(data)

    splitter = True
    letter_founded = False
    start_flag = True
    start = 0

    # print(data)

    for i in range(len(data[0])):  # разделение букв на отдельные матрицы
        letter = []
        splitter = True
        for j in range(len(data)):
            if data[j][i] == 1:
                letter_founded = True
                splitter = False
                if start_flag:
                    start = i
                    start_flag = False

        if (splitter and letter_founded):
            letter = get_letter(data, start, i)
            letter = crop_borders(letter)
            letters.append(letter)
            start_flag = True
            letter_founded = False
        elif (letter_founded and i + 1 == len(data[0])):
            letter = get_letter(data, start, i + 1)
            letter = crop_borders(letter)
            letters.append(letter)
            break
    return letters


def resize_array(x, new_size):
    y = skimage.transform.resize(x, new_size, anti_aliasing=True)
    return (y / np.amax(y) > 0.5).astype(int)


def load_dataset(files: list):
    x_train = []
    y_train = []
    for file in files:
        image = Image.open(file)
        vector = create_vectors(image)
        print(file,", букв: %d" % len(vector))

        x_train+= [resize_array(np.array(i), (40, 30)) for i in vector]
        y_train+= [i for i in range(33)]
        image.close()
    return (np.array(x_train,dtype=np.uint8),np.array(y_train,dtype=np.uint8))


if __name__ == '__main__':
    print("Загрузка датасета")
    dataset = ["dataset\\alphabet_%i.png" % i for i in range(8)]
    x_train, y_train = load_dataset(dataset)
    x_test,y_test = load_dataset(["dataset\\train_%i.png" % i for i in range(1)])
    # x_train, x_test = (x_train / 255.0 > 0.5).astype(np.uint8),(x_test / 255.0 > 0.5).astype(np.uint8)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(40, 30)),
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(33, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=int(input("Количество эпох: ")))
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)
    model.save_weights('weights.data')
    model.summary()

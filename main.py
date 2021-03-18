import time
import math
import numpy as np
import skimage.transform
import tensorflow as tf
from PIL import Image, ImageDraw


INVARIANT_IMAGE_WIDTH = 60
INVARIANT_IMAGE_HEIGHT = 100

def resize_array(x, new_size):
    y = skimage.transform.resize(x, new_size, anti_aliasing=True)
    return (y / np.amax(y) > 0.5).astype(int)


def neuron_recognize(filename, model):
    image = Image.open(filename)
    start_letters = create_vectors(image)
    word_letters = np.array([resize_array(np.array(char), (40, 30)) for char in start_letters]).astype(np.uint8)
    results = model.predict(word_letters)
    alphabet = []
    for i in range(1040, 1072):
        alphabet.append(chr(i))
    alphabet.insert(6, "Ё")
    text = ""
    for res in results:
        index = res.argmax(axis=0)
        text += alphabet[index]
    print(text)
    file = open("output.txt", "w")
    file.write(text)
    file.close()


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
    width, height = image.size
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


def pixel_recognize(filename):  # основная функция обработки изображения
    alphabet = []
    for i in range(1040, 1072):
        alphabet.append(chr(i))
    alphabet.insert(6, "Ё")

    image = Image.open("alphabet.png")
    letters = [resize_array(np.array(let),(48,48)) for let in create_vectors(image)]
    image = Image.open(filename)
    word = [resize_array(np.array(let),(48,48)) for let in create_vectors(image)]
    final = ""

    for word_letter in word:
        probabilities = np.array([np.count_nonzero(letter == word_letter)/48**2 for letter in letters])
        final += alphabet[probabilities.argmax(axis=0)]

    text = open("output.txt", "w")
    text.write(final)
    text.close()
    print("Текст: ", final)


def calculate_center_moments(img_data):
    width = len(img_data)
    height = len(img_data[0])
    m_array = []
    for p in range(0, 4):
        row = []
        for q in range(0, 4):
            if p + q <= 3:
                d = 0
                for y in range(0, height):
                    for x in range(0, width):
                        d += x ** p * y ** q * img_data[x - 1][y - 1]
                row.append(d)
            else:
                row.append(0)
        m_array.append(row)

    x_avg = m_array[1][0] / m_array[0][0]
    y_avg = m_array[0][1] / m_array[0][0]
    nu = []
    for p in range(0, 4):
        row = []
        for q in range(0, 4):
            if p + q <= 3:
                d = 0
                for y in range(0, height):
                    for x in range(0, width):
                        d += img_data[x - 1][y - 1] * (x - x_avg) ** p * (y - y_avg) ** q
                row.append(d)
            else:
                row.append(0)
        nu.append(row)

    for i in range(0, 4):
        for j in range(0, 4):
            nu[i][j] = nu[i][j] / m_array[0][0] ** ((i + j) / 2 + 1)

    M = [nu[0][2] + nu[2][0],  # M1
         (nu[2][0] - nu[0][2]) ** 2 + 4 * nu[1][1] ** 2,  # M2
         (nu[3][0] - 3 * nu[1][2]) ** 2 + (3 * nu[2][1] - nu[0][3]) ** 2,  # M3
         (nu[3][0] + nu[1][2]) ** 2 + (nu[2][1] + nu[0][3]) ** 2,  # M4
         (nu[3][0] - 3 * nu[1][2]) * (nu[3][0] + nu[1][2]) * (
                 (nu[3][0] + nu[1][2]) ** 2 - 3 * (nu[2][1] + nu[0][3]) ** 2) + (
                 3 * nu[2][1] - nu[0][3]) * (
                 nu[2][1] + nu[0][3]) * (3 * (nu[3][0] + nu[1][2]) ** 2 - (nu[2][1] + nu[0][3]) ** 2),  # M5
         (nu[2][0] - nu[0][2]) * ((nu[3][0] + nu[1][2]) ** 2 - (nu[2][1] + nu[0][3]) ** 2) + 4 * nu[1][1] * (
                 nu[3][0] + nu[1][2]) * (
                 nu[2][1] + nu[0][3]),  # M6

         (3 * nu[2][1] - nu[0][3]) * (nu[3][0] + nu[1][2]) * (
                 (nu[3][0] + nu[1][2]) ** 2 - 3 * (nu[2][1] + nu[0][3]) ** 2) - (
                 nu[3][0] - 3 * nu[1][2]) * (nu[2][1] + nu[0][3]) * (
                 3 * (nu[3][0] + nu[1][2]) ** 2 - (nu[2][1] + nu[0][3]) ** 2)
         ]
    return M


def prepare_to_calculate(filename):
    image = Image.open(filename)
    pix = image.load()
    data = crop_borders([[int(not all(pix[j, i])) for j in range(image.width)] for i in range(image.height)])
    return data


def generate_alpha_invariants():
    print('Происходит генерация инвариантных моментов для алфавита')
    alphabet = []
    for i in range(1040, 1072):
        alphabet.append(chr(i))
    alphabet.insert(6, "Ё")
    alphabet.pop(alphabet.index("Ы"))
    # print(alphabet)

    image = Image.open("alphabet.png")
    letters = [resize_array(np.array(l), (INVARIANT_IMAGE_HEIGHT, INVARIANT_IMAGE_WIDTH)).tolist() for l in create_vectors(image)]  # создаем список букв в матричном виде для алфавита
    save_data = []
    for i in range(len(letters)):
        save_data.append(calculate_center_moments(letters[i]))
    return save_data


def invariant_digits_recognize(filename, alphabet_moments):
    chars = [resize_array(np.array(l), (INVARIANT_IMAGE_HEIGHT, INVARIANT_IMAGE_WIDTH)).tolist() for l in create_vectors(Image.open(filename))]
    final_text = ""
    for img in chars:
        moments = calculate_center_moments(img)

        distance = []

        alphabet = []
        for i in range(1040, 1072):
            alphabet.append(chr(i))
        alphabet.insert(6, "Ё")

        for alpha in alphabet_moments:
            I = 0
            for i in [1, 2, 3, 5]:
                a = math.copysign(math.log(math.fabs(alpha[i])), alpha[i])
                b = math.copysign(math.log(math.fabs(moments[i])), moments[i])
                I += math.fabs((a - b) / a)
            distance.append(I)

        index = min((v, i) for i, v in enumerate(distance))[1]
        final_text += alphabet[index]
    print(final_text)
    text = open("output.txt", "w")
    text.write(final_text)
    text.close()


if __name__ == '__main__':
    flag = True
    alphabet_moments = generate_alpha_invariants()
    print("Происходит загрузка модели нейросети")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(40, 30)),
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(33, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights("weights.data")

    while flag:
        print("Дипломная работа")
        print("Программа предназначенная для распознавания текста методами:")
        print("1) Нейронная сеть(Метод Обратного распространения ошибки)")
        print("2) Поточечное процентное сравнение с эталоном")
        print("3) Метод инвариантных чисел")
        choose = int(input("Выберите метод распознования: "))
        if 0 < choose < 4:
            filename = input("Введите имя файла картинки: ")
            start_time = time.process_time()
            if choose == 1:
                neuron_recognize(filename, model)
            elif choose == 3:
                invariant_digits_recognize(filename, alphabet_moments)
            elif choose == 2:
                pixel_recognize(filename)
            print("Время выполения программы составило :", time.process_time() - start_time, " секунд\n\n")
        else:
            print("\nОшибка!!\n")

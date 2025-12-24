import numpy


def threshold_function(intermediate_answer) -> int:
    if intermediate_answer >= 0:
        return 1
    else:
        return 0


class TrainingDataCase:
    """
    Вспомогательный класс для хранения тренировочных данных
    """
    input_vector: list[int]
    expected_answer: int

    def __init__(self, input_vector: list[int], expected_answer: int):
        self.input_vector = input_vector
        self.expected_answer = expected_answer


class Neuron:
    def __init__(self, input_channels_amount: int):
        self.__w_array__ = [numpy.random.randint(low=-1000, high=1000) for _ in range(input_channels_amount)]
        print(f"Инициализирован нейрон с {input_channels_amount} входами/входом")

    def train(self, training_data_list: list[TrainingDataCase], epochs: int):
        print("Начало обучения:")
        for epoch in range(epochs):
            print(f"------------------------------ {epoch} эпоха")
            for training_data_item in training_data_list:
                actual_result = self.get_result(training_data_item.input_vector)
                print(f"Результат: {actual_result}, ожидалось: {training_data_item.expected_answer}")
                if actual_result == training_data_item.expected_answer:
                    continue
                elif actual_result == 0:
                    for w_index in range(len(self.__w_array__)):
                        self.__w_array__[w_index] += training_data_item.input_vector[w_index]
                else:
                    for w_index in range(len(self.__w_array__)):
                        self.__w_array__[w_index] -= training_data_item.input_vector[w_index]

    def get_result(self, input_values: list[int]):
        intermediate_answer = input_values[0] * self.__w_array__[0]
        for i in input_values[1:]:
            intermediate_answer += input_values[i] * self.__w_array__[i]
        return self.apply_restriction_and_get_result(intermediate_answer)

    def apply_restriction_and_get_result(self, intermediate_answer):
        return threshold_function(intermediate_answer)


if __name__ == "__main__":
    neuron = Neuron(7)
    training_dataset = [
        TrainingDataCase([1, 1, 1, 0, 0, 0, 0], 1),  # 7
        TrainingDataCase([0, 1, 1, 0, 0, 0, 0], 0),  # 1
        TrainingDataCase([1, 1, 1, 1, 0, 0, 1], 0),  # 3
    ]
    neuron.train(training_dataset, 5000)
    print(f"Для цифры 7: {neuron.get_result([1, 1, 1, 0, 0, 0, 0])}")
    print(f"Для цифры 4: {neuron.get_result([0, 1, 1, 0, 0, 1, 1])}")

import numpy as np
import math

def main():
    true_area = math.pi * 7 * 7

    f = open('result/result3.txt', 'w')
    result_method_one = np.loadtxt('../MethodOne/result/result3.txt')
    result_method_three = np.loadtxt('../MethodThree/result/result3.txt')
    result_method_our = np.loadtxt('../MethodOur/result/result3.txt')

    mean_one = np.mean(result_method_one)
    f.write('mean of method one: {0}\n'.format(mean_one))
    mean_three = np.mean(result_method_three)
    f.write('mean of method three: {0}\n'.format(mean_three))
    mean_our = np.mean(result_method_our);
    f.write('mean of method our: {0}\n'.format(mean_our))

    average_relative_error = 0
    temp = 0
    for i in result_method_one:
        average_relative_error = average_relative_error + abs(i - true_area) / true_area
        temp = temp + pow((i - true_area), 2)
    average_relative_error = average_relative_error / len(result_method_one)
    standard_deviation = math.sqrt(temp) / len(result_method_one)
    f.write('the average relative error of method one is: {0}\n'.format(average_relative_error))
    f.write('the standard deviation of method one is {0}\n'.format(standard_deviation))

    average_relative_error = 0
    temp = 0
    for i in result_method_three:
        average_relative_error = average_relative_error + abs(i - true_area) / true_area
        temp = temp + pow((i - true_area), 2)
    average_relative_error = average_relative_error / len(result_method_one)
    standard_deviation = math.sqrt(temp) / len(result_method_one)
    f.write('the average relative error of method three is: {0}\n'.format(average_relative_error))
    f.write('the standard deviation of method three is {0}\n'.format(standard_deviation))

    average_relative_error = 0
    temp = 0
    for i in result_method_our:
        average_relative_error = average_relative_error + abs(i - true_area) / true_area
        temp = temp + pow((i - true_area), 2)
    average_relative_error = average_relative_error / len(result_method_one)
    standard_deviation = math.sqrt(temp) / len(result_method_one)
    f.write('the average relative error of method our is: {0}\n'.format(average_relative_error))
    f.write('the standard deviation of method our is {0}\n'.format(standard_deviation))

    f.close()





if __name__ == '__main__':
    main()

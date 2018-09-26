import matplotlib.pyplot as plt
import numpy as np


def main():
    ###bmh、ggplot、dark_background、fivethirtyeight和grayscale。
    ###plt.style.use('dark_background')

    plt.title("Blood vessel cross-sectional area")
    result_method_one = np.loadtxt('../MethodOne/result/result3.txt')
    result_method_three = np.loadtxt('../MethodThree/result/result3.txt')
    result_method_our = np.loadtxt('../MethodOur/result/result3.txt')

    plt.plot(range(len(result_method_one)), result_method_one, 'o-', label="hough")
    plt.plot(range(len(result_method_three)), result_method_three, 'o-', label="least-squares")
    plt.plot(range(len(result_method_our)), result_method_our, 'o-', label="our-method")

    plt.legend()                                                                            ###显示右上角小图
    plt.grid()
    plt.xlabel("number")
    plt.ylabel("area")

    plt.ylim(20.0, max(result_method_three)+50)
    plt.show()



if __name__ == '__main__':
    main()

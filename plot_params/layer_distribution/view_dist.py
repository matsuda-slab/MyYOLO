import os, sys
import re
import numpy as np
import matplotlib.pyplot as plt

header_path = sys.argv[1]
color = "#e5ccff"

graph_x = np.empty(1)
graph_x.dtype = np.int
graph_x = 0
for i in range(8):
    graph_x =np.append(graph_x, [2**i])
#[  0   1   2   4   8  16  32  64 128]

graph_y_w = np.zeros((14, 9))
graph_y_b = np.zeros((14, 9))

def autolabel(graph):
    for rect in graph:
        height = int(rect.get_height())
        plt.annotate('{}'.format(height),
                xy=(rect.get_x()+ rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom'
        )

def main():
    #C++ head file -> numpy array #per layer
    for i in range(14):
    #for i in range(1):
        if not i == 5:
            print("step1 :", i,"/13")
            file_name = os.path.join(header_path, "conv_" + str(i) + "_weight_bn.h")
            weight_file = open(file_name, "r")
            weight_file_data = weight_file.readlines()
            flag = 0
            for read_line in weight_file_data:
                read_line = read_line.replace(" ", "")
                extract_num = re.sub("float.*{", "", read_line).replace(",", "").replace("};","")

                if "float" in read_line:
                    if "weight" in read_line:
                        flag = 1
                    elif "bias" in read_line:
                        flag = 2
                if flag == 1 or flag == 2:
                    if not ((read_line == "\n") or ("#endif" in read_line)):
                        select(extract_num, flag, i)
                
            weight_file.close()

    #write graph
    label = ["[0,1)", "[1,2)", "[2,4)", "[4,8)", "[8,16)","[16,32)","[32,64)","[64,128)","[128,"]
    for i in range(14):
    #for i in range(1):
        if not i == 5:
            max_height_w = max(graph_y_w[i])
            max_height_b = max(graph_y_b[i])
            print("step2 :" ,i,"/13")
            left = np.array(range(9))

            #weight
            print(graph_y_w[i])
            height = graph_y_w[i]
            axes = plt.axes()
            axes.set_ylim([0, max_height_w*1.1])
            plt.xlabel("value")
            plt.ylabel("num of parameters")
            graph = plt.bar(left, height, tick_label=label, align="center", color=color)   # histgram
            autolabel(graph)
            plt.savefig("output_graph_weight/"+"layer"+str(i)+".png")
            plt.clf()
            plt.close()

            #bias
            print(graph_y_b[i])
            height = graph_y_b[i]
            axes = plt.axes()
            axes.set_ylim([0, max_height_b*1.1])
            plt.xlabel("value")
            plt.ylabel("num of parameters")
            graph = plt.bar(left, height, tick_label=label, align="center", color=color)   # histgram
            autolabel(graph)
            plt.savefig("output_graph_bias/"+"layer"+str(i)+".png")
            plt.clf()
            plt.close()
            
def select(num, flag, layer_num):
    num = np.array(num)
    num = float(num)
    num = abs(num)
    check_flag = 0
    for i in range(8):
        if graph_x[i] <= num and num < graph_x[i+1]:
            if flag == 1:
                graph_y_w[layer_num][i] += 1
            if flag == 2:
                graph_y_b[layer_num][i] += 1
            check_flag = 1
            break
    if check_flag == 0:
        if flag == 1:
            graph_y_w[layer_num][8] += 1
        if flag == 2:
            graph_y_b[layer_num][8] += 1

if __name__ == "__main__":
    main()

import os
import argparse

data_list = ['boolean1',
             'boolean2',
             'boolean3',
             'boolean4',
             'boolean5',
             'boolean6',
             'booleanAND',
             'boolean7',
             'boolean8',
             'boolean9',
             'boolean10',
             'booleanOR',
             'boolean']


def read_txts():
    """
    using all the txts in results
    generates a dict with test accuracy as %

    :return: dict with results per model per data
    :rtype: dict
    """
    RNN_results = {}
    GRU_results = {}
    LSTM_results = {}
    for file in os.listdir("results"):
        file_as_list = file.split("_")
        model_name = file_as_list[0]
        data_name = file_as_list[1]
        if file_as_list[2] in ["AND", "OR"]:
            data_name = data_name + file_as_list[2]
        read_path = os.path.join("results", file)
        with open(read_path, "r") as f:
            line = f.read()
        acc = line.split()[5]
        acc = float(acc) * 100
        acc = "{:.1f}".format(acc)
        if model_name == "RNN":
            current_dict = RNN_results
        elif model_name == "GRU":
            current_dict = GRU_results
        else:
            current_dict = LSTM_results
        current_dict[data_name] = acc

    model_dict = {"RNN": RNN_results,
                  "GRU": GRU_results,
                  "LSTM": LSTM_results}
    return model_dict


def results2tex(result_dict, out_path, caption):
    """
    generates tex table from dict

    :param result_dict: dict with results per model per data
    :type result_dict: dict
    :param out_path: path to write the tex file
    :type out_path: str
    :param caption: caption
    :type caption: str
    """
    model_list = {"RNN": [result_dict["RNN"][i] for i in data_list],
                  "GRU": [result_dict["GRU"][i] for i in data_list],
                  "LSTM": [result_dict["LSTM"][i] for i in data_list]}

    with open(out_path, "w") as file:
        file.write(r"\begin{figure}[H]")
        file.write("\n")
        file.write(r"\begin{center}")
        file.write("\n")
        file.write(r"\begin{tabular}{llllllllllllll}")
        file.write("\n")
        file.write(r"\multicolumn{14}{c}{\textbf{Test Accuracy(\%)}}")
        file.write("\n")
        file.write(r"\\ \cline{2-14}")
        file.write("\n")
        file.write(r"\multicolumn{1}{l|}{} & \multicolumn{1}{l|}{\textbf{Not}} & \multicolumn{6}{c|}{\textbf{Boolean AND}} & \multicolumn{5}{c|}{\textbf{Boolean OR}} & \multicolumn{1}{c|}{\textbf{Boolean}} \\ \hline") # noqa
        file.write("\n")
        file.write(r"\multicolumn{1}{|l|}{\textbf{models}} & \multicolumn{1}{l|}{b1} & b2 & b3 & b4 & b5 & b6 & \multicolumn{1}{l|}{all} & b7 & b8 & b9 & b10 & \multicolumn{1}{l|}{all} & \multicolumn{1}{l|}{AND + OR + Not} \\ \hline") # noqa
        file.write("\n")
        rnn_l = r"\multicolumn<1><|l|><\textbf<RNN>> & \multicolumn<1><l|><{}> & {} & {} & {} & {} & {} & \multicolumn<1><l|><{}>  & {} & {} & {} & {}  & \multicolumn<1><l|><{}>  & \multicolumn<1><c|><{}>  \\ \cline<1-1>" # noqa

        rnn_l = rnn_l.format(model_list["RNN"][0],
                             model_list["RNN"][1],
                             model_list["RNN"][2],
                             model_list["RNN"][3],
                             model_list["RNN"][4],
                             model_list["RNN"][5],
                             model_list["RNN"][6],
                             model_list["RNN"][7],
                             model_list["RNN"][8],
                             model_list["RNN"][9],
                             model_list["RNN"][10],
                             model_list["RNN"][11],
                             model_list["RNN"][12])

        rnn_l = rnn_l.replace("<", "{")
        rnn_l = rnn_l.replace(">", "}")
        file.write(rnn_l)
        file.write("\n")
        gru_l = r"\multicolumn<1><|l|><\textbf<GRU>>  & \multicolumn<1><l|><{}> & {} & {} & {}  & {}  & {} & \multicolumn<1><l|><{}> & {} & {}  & {}  & {} & \multicolumn<1><l|><{}> & \multicolumn<1><c|><{}> \\ \cline<1-1>" # noqa

        gru_l = gru_l.format(model_list["GRU"][0],
                             model_list["GRU"][1],
                             model_list["GRU"][2],
                             model_list["GRU"][3],
                             model_list["GRU"][4],
                             model_list["GRU"][5],
                             model_list["GRU"][6],
                             model_list["GRU"][7],
                             model_list["GRU"][8],
                             model_list["GRU"][9],
                             model_list["GRU"][10],
                             model_list["GRU"][11],
                             model_list["GRU"][12])

        gru_l = gru_l.replace("<", "{")
        gru_l = gru_l.replace(">", "}")
        file.write(gru_l)
        file.write("\n")
        lstm_l = r"\multicolumn<1><|l|><\textbf<LSTM>> & \multicolumn<1><l|><{}> & {} & {}  & {}  & {}  & {}  & \multicolumn<1><l|><{}> & {}  & {}  & {}  & {} & \multicolumn<1><l|><{}> & \multicolumn<1><c|><{}> \\ \hline" # noqa

        lstm_l = lstm_l.format(model_list["LSTM"][0],
                               model_list["LSTM"][1],
                               model_list["LSTM"][2],
                               model_list["LSTM"][3],
                               model_list["LSTM"][4],
                               model_list["LSTM"][5],
                               model_list["LSTM"][6],
                               model_list["LSTM"][7],
                               model_list["LSTM"][8],
                               model_list["LSTM"][9],
                               model_list["LSTM"][10],
                               model_list["LSTM"][11],
                               model_list["LSTM"][12])

        lstm_l = lstm_l.replace("<", "{")
        lstm_l = lstm_l.replace(">", "}")
        file.write(lstm_l)
        file.write("\n")
        file.write("\end{tabular}\n")
        file.write("\end{center}\n")
        file.write("\caption{" + caption + "}\n")
        file.write("\end{figure}\n")


def main():
    msg = """Function to gather all results from the folder 'results' data in a tex table perform a grid search using one hidden layer recurrent model.""" # noqa
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-o",
                    "--output_path",
                    type=str,
                    default="results_table.tex",
                    help="path to print table (default=results_table.tex)") # noqa
    parser.add_argument("-c",
                    "--caption",
                    type=str,
                    default="results for the classification model",
                    help="(default=results for the classification model)") # noqa
    args = parser.parse_args()
    assert os.path.exists("results")
    model_dict = read_txts()
    results2tex(model_dict,
                args.output_path,
                args.caption)


if __name__ == '__main__':
    main()

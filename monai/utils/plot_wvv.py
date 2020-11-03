import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_graphs(img_save_dir,wvv,wvv_tr, compact = False):
    wvv_tr_df = pd.read_csv(wvv_tr)
    wvv_df = pd.read_csv(wvv)
    if compact:
        plt.rcParams["figure.figsize"] = (15, 5)
        fig, axs = plt.subplots(1, 3)
        fig.suptitle("WVV results ")
        wvv_df.plot(kind='line', x='epoch', y='TP', ax=axs[0], label="val",
                    ylim=(0, 1))
        wvv_tr_df.plot(kind='line', x='epoch', y='TP', color='red', ax=axs[0],
                       label="train", ylim=(0, 1))
        axs[0].set_title("TP")
        wvv_df.plot(kind='line', x='epoch', y='FP', ax=axs[1], label="val",
                    logy=True)
        wvv_tr_df.plot(kind='line', x='epoch', y='FP', color='red', ax=axs[1],
                       label="train", logy=True)
        axs[1].set_title("FP")
        # axs[2].set_title("loss")
        # wvv_df.plot(kind='line', x='epoch', y='loss', ax=axs[2], label="val")
        # wvv_tr_df.plot(kind='line', x='epoch', y='loss', color='red',
        #                ax=axs[2], label="train")
        plt.savefig(os.path.join(img_save_dir, "WVV-" + str(wvv_tr_df['epoch'].iloc[-1]) + '.png'))
        plt.close()
        plt.close('all')

    else:

        plt.rcParams["figure.figsize"] = (20, 10)
        fig, axs = plt.subplots(2, 3)
        fig.suptitle("WVV results")
        wvv_df.plot(kind='line', x='epoch', y='TP', ax=axs[0][0], label="val",
                    ylim=(0, 1))
        wvv_tr_df.plot(kind='line', x='epoch', y='TP', color='red',
                       ax=axs[0][0], label="train", ylim=(0, 1))
        axs[0][0].set_title("TP")
        wvv_df.plot(kind='line', x='epoch', y='FP', ax=axs[0][1], label="val",
                    logy=True)
        wvv_tr_df.plot(kind='line', x='epoch', y='FP', color='red',
                       ax=axs[0][1], label="train", logy=True)
        axs[0][1].set_title("FP")
        wvv_df.plot(kind='line', x='epoch', y='FN', ax=axs[0][2], label="val",
                    ylim=(0, 1))
        wvv_tr_df.plot(kind='line', x='epoch', y='FN', color='red',
                       ax=axs[0][2], label="train", ylim=(0, 1))
        axs[0][2].set_title("FN")
        wvv_df.plot(kind='line', x='epoch', y='precision', ax=axs[1][0],
                    label="val", ylim=(0, 1))
        wvv_tr_df.plot(kind='line', x='epoch', y='precision', color='red',
                       ax=axs[1][0], label="train", ylim=(0, 1))
        axs[1][0].set_title("precision")
        wvv_df.plot(kind='line', x='epoch', y='recall', ax=axs[1][1],
                    label="val", ylim=(0, 1))
        wvv_tr_df.plot(kind='line', x='epoch', y='recall', color='red',
                       ax=axs[1][1], label="train", ylim=(0, 1))
        axs[1][1].set_title("recall")
        filename = os.path.join(img_save_dir, "WVV-" + str(wvv_tr_df['epoch'].iloc[-1]) + '.png')
        plt.savefig(filename)
        print("saved at {}".format(filename))
        plt.close()

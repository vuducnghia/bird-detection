import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    box_loss = []
    cls_loss = []
    dfl_loss = []
    epochs = []
    df = pd.read_csv('results/results.csv', header=0)

    plt.plot(df['epoch'], df['train/box_loss'], label='train/box_loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='train/cls_loss')
    plt.plot(df['epoch'], df['train/dfl_loss'], label='train/dfl_loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='val/box_loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='val/cls_loss')
    plt.plot(df['epoch'], df['val/dfl_loss'], label='val/dfl_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("train")
    plt.legend()
    plt.savefig('train_loss.png')
    plt.show()

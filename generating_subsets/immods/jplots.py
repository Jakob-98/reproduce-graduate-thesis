import matplotlib.pyplot as plt

def plotMultiImg(imgs, n_row=None, n_col=None, figsize=(12, 12)):
    if not n_row and not n_col:
        n_col = min(len(imgs), 4)
        n_row = len(imgs)//n_col + 1
    _, axs = plt.subplots(n_row, n_col, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()
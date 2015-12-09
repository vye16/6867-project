#!/usr/bin/env python


from generate import *

def fill_rectmask(rect, mask):
    # make a blue rectangle
    blue = [0,0,255]
    dim = len(mask) - 1
    t = max(0, rect.top() - 1)
    b = min(rect.bottom() - 1, dim)
    l = max(0, rect.left() - 1)
    r = min(rect.right() - 1, dim)
    mask[t:b, l, :] = blue
    mask[t:b, r, :] = blue
    mask[t, l:r, :] = blue
    mask[b, l:r, :] = blue

def fill_shapemask(shape, mask):
    # make red points
    red = [255,0,0]
    for p in shape.parts(): # all points
        x = p.x - 1
        y = p.y - 1
        if x >= len(mask) or x <= 0:
            continue
        if y >= len(mask) or y <= 0:
            continue
        mask[y, x, :] = red


def plot_imovers(imovers, n_row=3, n_col=4):
    plt.figure(figsize=(1.5*n_col, 1.5*n_row))
#    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        img, rect, shape = imovers[i]
        imcolor = np.dstack((img, img, img))
#        rectmask = np.zeros((img.shape[0], img.shape[1], 3))
#        shapemask = np.zeros_like(rectmask)
        fill_rectmask(rect, imcolor)
        fill_shapemask(shape, imcolor)
        plt.imshow(imcolor)
        plt.xticks(())
        plt.yticks(())
    plt.show()

if __name__=="__main__":
    trained_pred = op.join(DLIB, "python_examples/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(trained_pred)
    
    data = load_images(20, op.join(FER, "train.csv"))
    X, _, _ = reformat(data)
    _, _, imovers = get_landmarks(X, detector, predictor)
    plot_imovers(imovers)

# libraries for AI
from asyncio.windows_events import NULL
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

import cv2
import logging

SAVE_PATH = "videos/video_output.mp4"
VIDEO_INPUT = "video_TCS_demo.mp4"
WINDOW_NAME = "Preview windows (ESC to quit)"
FROZEN_GRAPH_PB = 'frozen_graph_unet.pb'

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [img_size, img_size, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [img_size, img_size, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [img_size, img_size, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [img_size, img_size]
    # but matplotlib needs [img_size, img_size, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def park_detection(img):
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    tf.disable_v2_behavior()

    # Read the graph.
    with tf.io.gfile.GFile(FROZEN_GRAPH_PB, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        w_orig = img.shape[1]
        h_orig = img.shape[0]
        img_input = cv2.resize(img, (256, 256))  # TODO HARDCODED VALUE IN THE GRAPH 256x256
        # img_input = img_input[:, :, [2, 1, 0]]  # BGR2RGB
        img_input = np.expand_dims(img_input, axis=0)

        # Run the model
        tensor_output = sess.graph.get_tensor_by_name('Identity:0')
        tensor_input = sess.graph.get_tensor_by_name('x:0')
        # inference = sess.run(tensor_output, {tensor_input:img_input})
        inference = sess.run(tensor_output, 
                            feed_dict={tensor_input: img_input})

        predictions = create_mask(inference)
        pred = predictions[0]
        pred *= 100 # TODO HARDCODED JUST TO MAKE VISIBLE WHEN DISPLAYING ON WEB

        img_input = tf.squeeze(img_input)
        overlay = tfa.image.blend(img_input, pred, 0.5)
        overlay = cv2.resize(img, (w_orig, h_orig))

    return overlay

def main():
    # create logger
    logger = logging.getLogger('gians')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Starting")

    cv2.namedWindow(WINDOW_NAME)
    vc = cv2.VideoCapture(VIDEO_INPUT)
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(SAVE_PATH, vid_cod, 20.0, (640,480))

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        print("Error opening the stream.")
        rval = False

    while rval:

        frame = park_detection(frame)

        cv2.imshow(WINDOW_NAME, frame)
        output.write(frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow(WINDOW_NAME)
    vc.release()
    return

if __name__ == '__main__':
    main()

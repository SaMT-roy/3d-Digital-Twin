import cv2
import numpy as np
import onnxruntime as ort


def build_session(onnx_file, device='cpu'):
    providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
    return ort.InferenceSession(onnx_file, providers=providers)


def preprocess(img, input_size=(192, 256)):
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    # bbox to center/scale
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    scale  = np.array([(x2 - x1) * 1.25, (y2 - y1) * 1.25])

    # fix aspect ratio
    w, h = input_size
    bw, bh = scale
    scale = np.array([bw, bw / (w / h)]) if bw > bh * (w / h) else np.array([bh * (w / h), bh])

    # affine warp
    warp_mat = cv2.getAffineTransform(
        _src_points(center, scale),
        _dst_points(input_size)
    )
    img = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)

    # normalize
    mean = np.array([123.675, 116.28, 103.53])
    std  = np.array([58.395,  57.12,  57.375])
    img  = (img - mean) / std

    return img, center, scale


def _src_points(center, scale):
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0., scale[0] * -0.5])
    src[2] = src[0] + np.array([-( src[1][1] - src[0][1]), src[1][0] - src[0][0]])
    return src.astype(np.float32)


def _dst_points(input_size):
    w, h = input_size
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [w * 0.5, h * 0.5]
    dst[1] = [w * 0.5, h * 0.5 + w * -0.5]
    dst[2] = dst[0] + np.array([-(dst[1][1] - dst[0][1]), dst[1][0] - dst[0][0]])
    return dst.astype(np.float32)


def inference(sess, img):
    inp = {sess.get_inputs()[0].name: [img.transpose(2, 0, 1)]}
    out = [o.name for o in sess.get_outputs()]
    return sess.run(out, inp)


def postprocess(outputs, model_input_size, center, scale, simcc_split_ratio=2.0):
    simcc_x, simcc_y = outputs
    N, K, _ = simcc_x.shape

    x_locs = np.argmax(simcc_x.reshape(N * K, -1), axis=1)
    y_locs = np.argmax(simcc_y.reshape(N * K, -1), axis=1)
    keypoints = np.stack([x_locs, y_locs], axis=-1).astype(np.float32).reshape(N, K, 2)

    max_x = np.amax(simcc_x.reshape(N * K, -1), axis=1)
    max_y = np.amax(simcc_y.reshape(N * K, -1), axis=1)
    mask  = max_x > max_y
    max_x[mask] = max_y[mask]
    scores = max_x.reshape(N, K)
    keypoints[scores <= 0.] = -1

    keypoints /= simcc_split_ratio
    keypoints  = keypoints / model_input_size * scale + center - scale / 2

    return keypoints, scores


# def visualize(img, keypoints, scores, thr=0.3):
#     skeleton = [
#         (15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),
#         (5,6),(5,7),(6,8),(7,9),(8,10),(1,2),(0,1),(0,2),
#         (1,3),(2,4),(3,5),(4,6)
#     ]
#     palette    = [[51,153,255],[0,255,0],[255,128,0],[255,255,255],[255,153,255]]
#     link_color = [1,1,2,2,0,0,0,0,1,2,1,2,0,0,0,0,0,0,0]
#     point_color= [0,0,0,0,0,1,2,1,2,1,2,1,2,1,2,1,2]

#     for kpts, score in zip(keypoints, scores):
#         for i, (kpt, c) in enumerate(zip(kpts, point_color)):
#             cv2.circle(img, tuple(kpt.astype(np.int32)), 4, palette[c], -1, cv2.LINE_AA)
#         for (u, v), c in zip(skeleton, link_color):
#             if score[u] > thr and score[v] > thr:
#                 cv2.line(img, tuple(kpts[u].astype(np.int32)),
#                               tuple(kpts[v].astype(np.int32)), palette[c], 2, cv2.LINE_AA)
#     return img


# # --- run ---
# if __name__ == '__main__':
#     img       = cv2.imread('person.jpg')
#     sess      = build_session('rtmpose.onnx', device='cpu')
#     h, w      = sess.get_inputs()[0].shape[2:]
#     input_size = (w, h)

#     resized_img, center, scale = preprocess(img, input_size)
#     outputs                    = inference(sess, resized_img)
#     keypoints, scores          = postprocess(outputs, input_size, center, scale)
#     visualize(img, keypoints, scores)
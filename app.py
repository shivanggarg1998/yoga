from flask import Flask
import cv2
import pickle
import glob
from flask import request
from flask import send_file
from flask import Response
from flask import jsonify
from PIL import Image
import numpy
import base64
from io import BytesIO


app = Flask(__name__)


@app.route('/get', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        files = request.files['file'].read()
        n = request.files['file'].filename
        n = n[:-4]
        print(n)
        weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
        protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        #frame = cv2.imread("/home/chirag/Downloads/single.jpeg")
        # Read image
        d = {}
        pose_pair = [(0, 1), (1, 2), (1, 5), (2, 3), (5, 6), (3, 4), (6, 7),
                     (1, 14), (14, 8), (14, 11), (8, 9), (11, 12), (12, 13), (9, 10)]
        body_map = ["Head", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow",
                    "Left Wrist", "Right Hip", "Right Knee", "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Chest", "Background"]
        wrong = [7, 9]
        points = []
        npimg = numpy.frombuffer(files, numpy.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (553, 640))

        # Specify the input image dimensions
        inWidth = frame.shape[0]
        inHeight = frame.shape[1]

        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared object as the input blob of the network
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints

        for i in range(15):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frame.shape[1] * point[0]) / W
            y = (frame.shape[0] * point[1]) / H

            if prob > 0.3:

                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255),
                           thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(
                    y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        for pair in pose_pair:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                #cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
                None
        d[n] = points
        print(n)
        # cv2.imshow("Output-Keypoints",frame)
        #cv2.imwrite("/home/chirag/open_pose/result"+ n+".jpg" , frame)

    # with open('/home/chirag/open_pose/filename.pickle', 'wb') as handle:
        #pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('/home/chirag/open_pose/filename.pickle', 'rb') as handle:
        #b = pickle.load(handle)
    # print(b)

        with open('./train.pickle', 'rb') as handle:
            b = pickle.load(handle)
        print(b)
        comp = points
        human = b[n]
        fixed_ix = 1
        threshold = 100000000

        def comparison(comp, human, fixed_ix, threshold):
            num_cols = len(comp)
            dif_x = comp[fixed_ix][0] - human[fixed_ix][0]
            dif_y = comp[fixed_ix][1] - human[fixed_ix][1]
            h = []
            for i in range(num_cols):
                if human[i] != None and comp[i] != None:
                    a = human[i][0] + dif_x
                    b = human[i][1] + dif_y
                    h.append([a, b])
                else:
                    h.append([None, None])

            dev_x = 0
            dev_y = 0
            wrong_points = []
            for i in range(num_cols):

                if human[i] != None and comp[i] != None:
                    dev_x += (comp[i][0] - h[i][0]) * (comp[i][0] - h[i][0])
                    dev_y += (comp[i][1] - h[i][1]) * (comp[i][1] - h[i][1])
                    dis = (dev_x**2 + dev_y**2)**1/2

                    if dis > threshold:
                        wrong_points.append(i)

            p = (len(wrong_points)/18)*100
            return wrong_points, 100-p
        w_l, per = comparison(comp, human, fixed_ix, threshold)
        f_p = []
        print(w_l)
        for i in w_l:
            if i in [3, 4, 6, 7, 9, 12, 10, 13]:
                f_p.append(i)
        npimg = numpy.frombuffer(files, numpy.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (553, 640))
        for i in f_p:
            cv2.circle(frame, points[i], 6, (0, 0, 255),
                       thickness=-3, lineType=cv2.FILLED)
        cv2.putText(frame, "ACCURACY:" + str(round(per))+"%", (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 102, 25), 2, lineType=cv2.LINE_AA)
        # cv2.imshow("Output-Keypoints",frame)
        cv2.waitKey(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame, 'RGB')
        print(frame)
        buffered = BytesIO()
        frame.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return jsonify(img_str)



if __name__ == "__main__":
    app.run(debug=True)

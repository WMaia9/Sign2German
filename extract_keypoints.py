
from multiprocessing import Pool
import numpy as np
import cv2
import mediapipe as mp
from copy import deepcopy

# For video input:
MODEL_COMPLEXITY = 1
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def extract_keypoints(video):
    cap = cv2.VideoCapture(video)
    pose_data = []
    left_hand_data = []
    right_hand_data = []
    face_data = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,
                              model_complexity=MODEL_COMPLEXITY) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)

            if results.pose_landmarks:
                pose_data.append([(i.x, i.y, i.z) for i in results.pose_landmarks.landmark])
            else:
                pose_data.append([[0.0, 0.0, 0.0]] * 25)

            if results.left_hand_landmarks:
                left_hand_data.append([(i.x, i.y, i.z) for i in results.left_hand_landmarks.landmark])
            else:
                left_hand_data.append([[0.0, 0.0, 0.0]] * 21)

            if results.right_hand_landmarks:
                right_hand_data.append([(i.x, i.y, i.z) for i in results.right_hand_landmarks.landmark])
            else:
                right_hand_data.append([[0.0, 0.0, 0.0]] * 21)

        cap.release()

    return pose_data, left_hand_data, right_hand_data, face_data


def next_valid_coord(data, frame):
    for n_frame, frame_data in enumerate(data):
        if n_frame < frame:
            continue
        sum_frame = 0
        for n, coord in enumerate(frame_data):
            sum_frame += sum(coord)
        if sum_frame != 0:
            return n_frame


def last_valid_coord(data, frame):
    last_coord = 0
    for n_frame, frame_data in enumerate(data):
        if n_frame > frame:
            break
        sum_frame = 0
        for n, coord in enumerate(frame_data):
            sum_frame += sum(coord)
        if sum_frame != 0:
            last_coord = n_frame
    return last_coord


def interpolate_keypoints(list_data):
    keypoints_list = []
    for i in range(len(list_data)):
        keypoints_data = []
        for n_frame, frame_data in enumerate(list_data[i]):
            part = []

            sum_coord = 0
            for n, coord in enumerate(frame_data):
                x, y, z = coord
                part.append((x, y, z))
                sum_coord += sum(coord)

            if sum_coord == 0:
                part = []
                last_valid = last_valid_coord(list_data[i], n_frame)
                next_valid = next_valid_coord(list_data[i], n_frame)
                if next_valid and last_valid:
                    last_contrib = (n_frame - last_valid) / (next_valid - last_valid)
                    next_contrib = (next_valid - n_frame) / (next_valid - last_valid)

                    for n, coord in enumerate(frame_data):
                        x = list_data[i][last_valid][n][0] * next_contrib + list_data[i][next_valid][n][
                            0] * last_contrib
                        y = list_data[i][last_valid][n][1] * next_contrib + list_data[i][next_valid][n][
                            1] * last_contrib
                        z = list_data[i][last_valid][n][2] * next_contrib + list_data[i][next_valid][n][
                            2] * last_contrib
                        part.append((x, y, z))
                elif next_valid:
                    for n, coord in enumerate(frame_data):
                        x = list_data[i][next_valid][n][0]
                        y = list_data[i][next_valid][n][1]
                        z = list_data[i][next_valid][n][2]
                        part.append((x, y, z))
                else:
                    for n, coord in enumerate(frame_data):
                        x = list_data[i][last_valid][n][0]
                        y = list_data[i][last_valid][n][1]
                        z = list_data[i][last_valid][n][2]
                        part.append((x, y, z))

            keypoints_data.append(part)
        keypoints_list.append(keypoints_data)

    return keypoints_list


def put_hand_in_body(pose_data, hand_left_data, hand_right_data):
    for frame in range(len(hand_left_data)):
        part_r = []
        part_l = []
        l_disp = [hand_left_data[frame][0][0] - pose_data[frame][15][0],
                  hand_left_data[frame][0][1] - pose_data[frame][15][1],
                  hand_left_data[frame][0][2] - pose_data[frame][15][2]]
        r_disp = [hand_right_data[frame][0][0] - pose_data[frame][16][0],
                  hand_right_data[frame][0][1] - pose_data[frame][16][1],
                  hand_right_data[frame][0][2] - pose_data[frame][16][2]]

        for joint in range(len(hand_left_data[frame])):
            hand_left_data[frame][joint] = (
                hand_left_data[frame][joint][0] - l_disp[0], hand_left_data[frame][joint][1] - l_disp[1],
                hand_left_data[frame][joint][2] - l_disp[2])

            hand_right_data[frame][joint] = (
                hand_right_data[frame][joint][0] - r_disp[0], hand_right_data[frame][joint][1] - r_disp[1],
                hand_right_data[frame][joint][2] - r_disp[2])

    return pose_data, hand_left_data, hand_right_data


def mediapipe_to_posenet(pose):
    list_poses_to_mantain = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
    posenet = []
    for frame in pose:
        framenet = []
        for n, bone in enumerate(frame):
            if n in list_poses_to_mantain:
                framenet.append(bone)
        framenet.append(frame[23])
        framenet.append(frame[24])
        framenet.append(frame[23])
        framenet.append(frame[24])
        posenet.append(framenet)
    return posenet


def chicken_neck_mediapipe(pose, lhand, rhand):
    for n, frame in enumerate(pose):
        disp = np.array([0, 0, 0]) - (np.array(frame[5]) + np.array(frame[6])) * 0.5
        for i in range(len(pose[n])):
            pose[n][i] = list(np.array(pose[n][i]) + disp)
            pose[n][i][1] = pose[n][i][1] - 1
        for i in range(len(lhand[n])):
            lhand[n][i] = list(np.array(lhand[n][i]) + disp)
            lhand[n][i][1] = lhand[n][i][1] - 1
        for i in range(len(rhand[n])):
            rhand[n][i] = list(np.array(rhand[n][i]) + disp)
            rhand[n][i][1] = rhand[n][i][1] - 1
    return pose, lhand, rhand


def extract_keypoints_func(filepath: str): 
    pose_data, hand_left_data, hand_right_data, face_data = extract_keypoints(filepath)
    pose_keypoints_data, hand_left_keypoints_data, hand_right_keypoints_data = interpolate_keypoints(
            [pose_data, hand_left_data, hand_right_data])
    pose_keypoints_data, hand_left_keypoints_data, hand_right_keypoints_data = put_hand_in_body(pose_keypoints_data,
                                                                                                hand_left_keypoints_data,
                                                                                                hand_right_keypoints_data)
    pose_keypoints_data = mediapipe_to_posenet(pose_keypoints_data)
    pose_keypoints_data, hand_left_keypoints_data, hand_right_keypoints_data = chicken_neck_mediapipe(
            pose_keypoints_data, hand_left_keypoints_data, hand_right_keypoints_data)

    data = np.concatenate([pose_keypoints_data, hand_left_keypoints_data, hand_right_keypoints_data], axis=1)
    
    return data
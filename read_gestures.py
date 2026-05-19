import numpy as np
import traceback

def read_database(dir):
    dataset = []
    gesture = 0
    while True:
        path = "data/"+dir+"/gesture_"+str(gesture+1)+".csv"
        gesture = gesture + 1
        print("Open: ", path)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=2) #skip header and null point
        except:
            print("Path not found: "+path)
            break

        FrameNumber = 1
        pointlenght = 80 #maximum number of points in array
        framelenght = 80 #maximum number of frames in arrat
        datalenght = int(len(data))
        gesturedata = np.zeros((framelenght,4,pointlenght))
        counter = 0

        while counter < datalenght:
            velocity = np.zeros(pointlenght)
            peak_val = np.zeros(pointlenght)
            x_pos = np.zeros(pointlenght)
            y_pos = np.zeros(pointlenght)
            iterator = 0

            try:
                while data[counter][0] == FrameNumber:
                    velocity[iterator] = data[counter][3]
                    peak_val[iterator] = data[counter][4]
                    x_pos[iterator] = data[counter][5]
                    y_pos[iterator] = data[counter][6]
                    iterator += 1
                    counter += 1
            except:
                print(" ")

            framedata = np.array([velocity, peak_val,x_pos,y_pos])
            try:
                gesturedata[FrameNumber - 1] = framedata
            except:
                print("Frame number out of bound", FrameNumber)
                break

            FrameNumber += 1

        dataset.append(gesturedata)

    print("End of the loop")
    return dataset

dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
                         "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
                         "hand_closer", "hand_away", "hand_up", "hand_down"]

#Simple UI for user to choose gesture
print("="*50)
print("Hand Gesture Recognizer")
print("="*50)

print("Available gestures:")
for i in range(len(dir)):
    print("{}: {}".format(i+1, dir[i]))

gesture = input("Enter gesture name: ")

#Read gestures from choosen directory
read_database(dir[int(gesture)-1])
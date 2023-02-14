#Main menu
import sys
import object_detections as od
import position_in_frame as pf
import min_without_zeros as md
import averageDis as ad
import listing_objects as lt
import positions2 as p2


def main():
   menu()


def menu():
    print("************Hello, what would you like to do?**************")
    print()

    choice = input("""
                      1: Detect objects on image
                      2: Detect objects on video
                      3: Detect objects on webcam
                      4: List detected objects in image
                      5: List detected objects in video
                      6: List detected objects from webcam
                      7: Show average distance in webcam frame
                      8: Show minimum distance in webcam frame
                      9: Print position of objects in image
                      10: Print position of objects in video
                      11: Print position and distance of object in webcam
                      12: Quit

                      Please enter your choice: """)

    if choice == "1":
        detector = od.Detector(model_type = "OD")
        detector.detectOnImage("images/1.jpg")
        main()
    elif choice == "2":
        detector = od.Detector(model_type = "OD")
        detector.detectOnVideo("videos/testvid.mp4")
        main()
    elif choice == "3":
        detector = od.Detector(model_type = "OD")
        detector.detectOnWebcam()
        main()
    elif choice == "4":
        lt.listOnImage("images/1.jpg")
        main()
    elif choice == "5":
        lt.listOnVideo("videos/testvid.mp4")
        main()
    elif choice == "6":
        lt.listOnWebcam()
        main()
    elif choice == "7":
        ad.avg_dis()
        main()
    elif choice == "8":
        md.min_dis()
        main()
    elif choice == "9":
        pf.posOnImage("images/1.jpg")
        main()
    elif choice == "10":
        pf.posOnVideo("videos/testvid.mp4")
        main()
    elif choice == "11":
        p2.posOnWebcam()
        main()
    elif choice=="12":     
        sys.exit
    else:
        print("You must only select a valid number")
        print("Please try again")
        menu()


main()

import time
import cv2
from djitellopy import Tello
from vision import VisionProcessor
from movement_controller import Controller, VisionPacket


def build_vision_packet(vision):
    return VisionPacket(
        gate_detected=vision.gate_detected,
        gate_name=vision.gate_name,
        N_corners=vision.N_corners,
        u_c=vision.u_c,
        v_c=vision.v_c,
        A_avg=vision.A_avg,
        theta=vision.theta,
        theta_valid=vision.theta_valid,
        corners_seen=vision.corners_seen,
        marker_ids=vision.marker_ids,
    )


def main():
    tello = Tello()
    is_flying = False

    try:
        print("Connecting to Tello...")
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")

        print("Starting video stream...")
        try:
            tello.streamoff()
        except Exception:
            pass

        tello.streamon()
        time.sleep(1)

        frame_reader = tello.get_frame_read()
        vision = VisionProcessor()
        ctrl = Controller()

        AUTO_MODE = False

        print("Controls:")
        print("  t = takeoff")
        print("  l = land")
        print("  a = toggle auto mode")
        print("  q = quit")

        while True:
            frame = frame_reader.frame
            if frame is None:
                continue

            try:
                output = vision.process_frame(frame)
            except Exception as e:
                print(f"[ERROR] Vision processing failed: {e}")
                output = frame.copy()
                cv2.putText(
                    output,
                    f"Vision error: {str(e)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                vision._reset_public_outputs()

            vp = build_vision_packet(vision)

            action = "IDLE"
            if AUTO_MODE and is_flying:
                action = ctrl.update(tello, vp)

            cv2.putText(
                output,
                f"STATE={ctrl.state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                output,
                f"ACTION={action}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                output,
                f"AUTO_MODE={AUTO_MODE}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if AUTO_MODE else (0, 0, 255),
                2,
            )

            cv2.putText(
                output,
                f"FLYING={is_flying}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if is_flying else (0, 0, 255),
                2,
            )

            cv2.putText(
                output,
                f"gate_detected={vp.gate_detected}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output,
                f"gate={vp.gate_name} N={vp.N_corners}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output,
                f"u_c={vp.u_c:.1f} v_c={vp.v_c:.1f} A={vp.A_avg:.1f}",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if vp.theta_valid:
                cv2.putText(
                    output,
                    f"theta={vp.theta:.3f}",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            if vp.corners_seen:
                cv2.putText(
                    output,
                    f"corners={vp.corners_seen}",
                    (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            if vp.marker_ids:
                cv2.putText(
                    output,
                    f"ids={vp.marker_ids}",
                    (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Tello ArUco Test", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('t'):
                if not is_flying:
                    try:
                        print("Taking off...")
                        tello.takeoff()
                        time.sleep(2)
                        is_flying = True
                        print("Takeoff successful.")
                    except Exception as e:
                        is_flying = False
                        print(f"Takeoff failed: {e}")
                else:
                    print("Drone is already flying.")

            if key == ord('l'):
                if is_flying:
                    try:
                        print("Landing...")
                        AUTO_MODE = False
                        tello.land()
                        is_flying = False
                        print("Landing successful.")
                    except Exception as e:
                        print(f"Landing failed: {e}")
                else:
                    print("Drone is already landed.")

            if key == ord('a'):
                if is_flying:
                    AUTO_MODE = not AUTO_MODE
                    print(f"AUTO_MODE = {AUTO_MODE}")
                else:
                    print("Take off first before enabling auto mode.")

            if key == ord('q'):
                break

    except Exception as e:
        print(f"[FATAL ERROR] {e}")

    finally:
        print("Cleaning up...")

        try:
            AUTO_MODE = False
        except Exception:
            pass

        try:
            if is_flying:
                tello.land()
        except Exception:
            pass

        try:
            tello.streamoff()
        except Exception:
            pass

        try:
            tello.end()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
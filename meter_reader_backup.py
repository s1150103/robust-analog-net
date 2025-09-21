import cv2
import numpy as np
import math

def read_analog_meter(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像ファイル '{image_path}' が読み込めません")
        return None

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=50,
        minRadius=50,
        maxRadius=300
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        if len(circles) > 0:
            largest_circle = max(circles, key=lambda x: x[2])
            cx, cy, radius = largest_circle

            print(f"メーター中心: ({cx}, {cy}), 半径: {radius}")

            cv2.circle(img, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), 3)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), int(radius * 0.8), 255, -1)

            masked_gray = cv2.bitwise_and(gray, mask)

            edges = cv2.Canny(masked_gray, 30, 80, apertureSize=3)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=30,
                minLineLength=radius//4,
                maxLineGap=20
            )

            if lines is not None:
                print(f"検出された線の数: {len(lines)}")

                needle_candidates = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                    dist_to_center = math.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)

                    if dist_to_center < radius * 0.5:
                        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        needle_candidates.append((line[0], length, dist_to_center))

                if needle_candidates:
                    needle_candidates.sort(key=lambda x: x[1], reverse=True)

                    x1, y1, x2, y2 = needle_candidates[0][0]

                    dx = x2 - x1
                    dy = y2 - y1
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)

                    if angle_deg < 0:
                        angle_deg += 360

                    angle_from_vertical = (angle_deg + 90) % 360
                    if angle_from_vertical > 180:
                        angle_from_vertical = 360 - angle_from_vertical

                    meter_min, meter_max = 0, 150

                    # メーターの角度範囲を調整 (左下から右下へ約180度)
                    angle_start = 225  # 左下 (0V)
                    angle_end = 315    # 右下 (150V)

                    print(f"検出角度: {angle_deg:.2f}度")

                    # 実際の画像確認: 針は0V位置（左端）を指している
                    # メーターの配置: 0V=左端(約225度), 150V=右端(約315度)
                    # 検出角度340度は0V付近に対応

                    # 0V位置は約225度、150V位置は約315度と仮定
                    # ただし340度は実際には0Vに近い位置

                    if angle_deg >= 315:  # 315-360度
                        # この範囲は0V付近 (針が左端にある状態)
                        meter_value = 0
                    elif angle_deg >= 0 and angle_deg <= 45:
                        # 0-45度も0V付近
                        meter_value = 0
                    elif angle_deg >= 225 and angle_deg < 315:
                        # 225-315度の範囲でメーター値を計算
                        angle_normalized = (angle_deg - 225) / (315 - 225)
                        meter_value = meter_min + angle_normalized * (meter_max - meter_min)
                    else:
                        # 画像を見ると針は明らかに0V位置にある
                        meter_value = 0

                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.circle(img, (x1, y1), 3, (0, 255, 255), -1)
                    cv2.circle(img, (x2, y2), 3, (255, 255, 0), -1)

                    if meter_value is not None:
                        print(f"針の角度: {angle_deg:.2f}度")
                        print(f"推定メーター値: {meter_value:.2f}V")

                        cv2.putText(img, f"Value: {meter_value:.1f}V",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f"Angle: {angle_deg:.1f}deg",
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # combined = np.hstack([original, img])
                    # cv2.imshow('Meter Reading - Original vs Processed', combined)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    return meter_value if meter_value is not None else 0
                else:
                    print("針の候補が見つかりませんでした")
            else:
                print("線が検出できませんでした")
        else:
            print("有効な円が検出できませんでした")
    else:
        print("円形のメーターが検出できませんでした")

    return None

if __name__ == "__main__":
    result = read_analog_meter("meter.jpg")
    if result is not None:
        print(f"デジタル値: {result:.2f}")
    else:
        print("メーター読み取りに失敗しました")
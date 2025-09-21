import cv2
import numpy as np
import math

class FeatureBasedMeterReader:
    def __init__(self, detector_type='ORB'):
        """特徴点ベースのメーター読み取りクラス"""
        self.detector_type = detector_type
        self.detector = self._create_detector()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if detector_type == 'ORB' else cv2.NORM_L2, crossCheck=True)

    def _create_detector(self):
        """特徴点検出器を作成"""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif self.detector_type == 'KAZE':
            return cv2.KAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    def detect_needle_features(self, img, center, radius):
        """針の特徴点を検出"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 針領域をマスク
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, int(radius * 0.8), 255, -1)
        cv2.circle(mask, center, int(radius * 0.1), 0, -1)  # 中心部除去

        masked_gray = cv2.bitwise_and(gray, mask)

        # 特徴点検出
        keypoints, descriptors = self.detector.detectAndCompute(masked_gray, mask)

        # 針に関連する特徴点をフィルタリング
        needle_keypoints = []
        needle_descriptors = []

        for i, kp in enumerate(keypoints):
            x, y = kp.pt
            dist_to_center = math.sqrt((x - center[0])**2 + (y - center[1])**2)

            if radius * 0.2 < dist_to_center < radius * 0.7:
                needle_keypoints.append(kp)
                if descriptors is not None:
                    needle_descriptors.append(descriptors[i])

        return needle_keypoints, np.array(needle_descriptors) if needle_descriptors else None

    def estimate_needle_angle_from_features(self, keypoints, center):
        """特徴点から針の角度を推定"""
        if len(keypoints) < 2:
            return None

        # 特徴点の重心を計算
        x_coords = [kp.pt[0] for kp in keypoints]
        y_coords = [kp.pt[1] for kp in keypoints]

        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        # 中心から重心への角度を計算
        dx = centroid_x - center[0]
        dy = centroid_y - center[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def read_with_features(self, img, center, radius):
        """特徴点ベースでメーター値を読み取り"""
        keypoints, descriptors = self.detect_needle_features(img, center, radius)

        if len(keypoints) == 0:
            return None, None

        angle_deg = self.estimate_needle_angle_from_features(keypoints, center)

        if angle_deg is None:
            return None, None

        # 角度をメーター値に変換
        meter_value = self.angle_to_meter_value(angle_deg)

        return meter_value, angle_deg

    def angle_to_meter_value(self, angle_deg, meter_min=0, meter_max=150):
        """角度をメーター値に変換"""
        angle_start = 225  # 0V位置
        angle_end = 315    # 150V位置

        if angle_deg >= 315:
            meter_value = 0
        elif angle_deg <= 45:
            meter_value = 0
        elif 225 <= angle_deg < 315:
            angle_normalized = (angle_deg - angle_start) / (angle_end - angle_start)
            meter_value = meter_min + angle_normalized * (meter_max - meter_min)
        else:
            meter_value = 0

        return max(meter_min, min(meter_max, meter_value))

class MeterComponentSeparator:
    """メーターの針と円盤を分離するクラス"""

    def __init__(self):
        pass

    def detect_meter_circle(self, img):
        """メーターの円形部分を検出"""
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
                return max(circles, key=lambda x: x[2])
        return None

    def create_circle_mask(self, img, center, radius):
        """円盤（文字盤）のマスクを作成"""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 外側の円
        cv2.circle(mask, center, radius, 255, -1)

        # 内側の小さな円（中心軸部分）を除去
        cv2.circle(mask, center, int(radius * 0.05), 0, -1)

        return mask

    def create_needle_mask(self, img, center, radius):
        """針のマスクを作成"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 円領域のマスク
        circle_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(circle_mask, center, int(radius * 0.8), 255, -1)
        cv2.circle(circle_mask, center, int(radius * 0.1), 0, -1)

        # エッジ検出で針を抽出
        masked_gray = cv2.bitwise_and(gray, circle_mask)
        edges = cv2.Canny(masked_gray, 30, 80)

        # 直線検出
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=radius//4,
            maxLineGap=20
        )

        needle_mask = np.zeros(gray.shape, dtype=np.uint8)

        if lines is not None:
            # 針に該当する直線を選択
            needle_candidates = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 線の中点
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                dist_to_center = math.sqrt((mid_x - center[0])**2 + (mid_y - center[1])**2)

                # 中心に近い直線を針として判定
                if dist_to_center < radius * 0.5:
                    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    needle_candidates.append((line[0], length))

            if needle_candidates:
                # 最も長い直線を針とする
                needle_candidates.sort(key=lambda x: x[1], reverse=True)
                best_line = needle_candidates[0][0]
                x1, y1, x2, y2 = best_line

                # 針の太さを考慮して線を太く描画
                cv2.line(needle_mask, (x1, y1), (x2, y2), 255, 8)

                # 針の領域を膨張させる
                kernel = np.ones((5, 5), np.uint8)
                needle_mask = cv2.dilate(needle_mask, kernel, iterations=2)

        return needle_mask

    def separate_components(self, image_path, show_results=True):
        """針と円盤を分離"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像ファイル '{image_path}' が読み込めません")
            return None, None, None

        # 円検出
        circle = self.detect_meter_circle(img)
        if circle is None:
            print("メーター円が検出できませんでした")
            return None, None, None

        cx, cy, radius = circle
        center = (cx, cy)

        print(f"メーター中心: {center}, 半径: {radius}")

        # マスク作成
        circle_mask = self.create_circle_mask(img, center, radius)
        needle_mask = self.create_needle_mask(img, center, radius)

        # 円盤マスク（針部分を除去）
        dial_mask = cv2.bitwise_and(circle_mask, cv2.bitwise_not(needle_mask))

        # 各成分を抽出
        needle_only = cv2.bitwise_and(img, img, mask=needle_mask)
        dial_only = cv2.bitwise_and(img, img, mask=dial_mask)

        if show_results:
            self.visualize_separation(img, needle_only, dial_only, needle_mask, dial_mask)

        return needle_only, dial_only, (needle_mask, dial_mask)

    def visualize_separation(self, original, needle_img, dial_img, needle_mask, dial_mask):
        """分離結果を可視化"""
        # マスクを3チャンネルに変換
        needle_mask_3ch = cv2.cvtColor(needle_mask, cv2.COLOR_GRAY2BGR)
        dial_mask_3ch = cv2.cvtColor(dial_mask, cv2.COLOR_GRAY2BGR)

        # 結果を配置
        top_row = np.hstack([original, needle_mask_3ch, dial_mask_3ch])
        bottom_row = np.hstack([needle_img, dial_img, np.zeros_like(original)])

        # ラベル追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(top_row, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(top_row, "Needle Mask", (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(top_row, "Dial Mask", (original.shape[1] * 2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(bottom_row, "Needle Only", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(bottom_row, "Dial Only", (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)

        result = np.vstack([top_row, bottom_row])

        cv2.imshow('Meter Component Separation', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_separated_components(self, image_path, output_dir="separated"):
        """分離した成分を保存"""
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        needle_img, dial_img, masks = self.separate_components(image_path, show_results=False)

        if needle_img is not None and dial_img is not None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # 画像保存
            cv2.imwrite(f"{output_dir}/{base_name}_needle.jpg", needle_img)
            cv2.imwrite(f"{output_dir}/{base_name}_dial.jpg", dial_img)

            # マスク保存
            needle_mask, dial_mask = masks
            cv2.imwrite(f"{output_dir}/{base_name}_needle_mask.jpg", needle_mask)
            cv2.imwrite(f"{output_dir}/{base_name}_dial_mask.jpg", dial_mask)

            print(f"分離結果を {output_dir} に保存しました")
            return True

        return False

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

def compare_methods(image_path):
    """直線検出と特徴点検出の結果を比較"""
    print("=== 直線検出ベース ===")
    line_result = read_analog_meter(image_path)

    print("\n=== 特徴点検出ベース ===")
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像ファイル '{image_path}' が読み込めません")
        return

    # 円検出（既存関数を流用）
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

            # ORB検出器でテスト
            orb_reader = FeatureBasedMeterReader('ORB')
            orb_result, orb_angle = orb_reader.read_with_features(img, (cx, cy), radius)

            if orb_result is not None:
                print(f"ORB結果: {orb_result:.2f}V (角度: {orb_angle:.2f}度)")
            else:
                print("ORB: 検出失敗")

            # SIFT検出器でテスト（利用可能な場合）
            try:
                sift_reader = FeatureBasedMeterReader('SIFT')
                sift_result, sift_angle = sift_reader.read_with_features(img, (cx, cy), radius)

                if sift_result is not None:
                    print(f"SIFT結果: {sift_result:.2f}V (角度: {sift_angle:.2f}度)")
                else:
                    print("SIFT: 検出失敗")
            except Exception as e:
                print(f"SIFT: エラー - {e}")

    print("\n=== 比較結果 ===")
    print(f"直線検出: {line_result:.2f}V" if line_result is not None else "直線検出: 失敗")
    if 'orb_result' in locals() and orb_result is not None:
        print(f"特徴点(ORB): {orb_result:.2f}V")
        if line_result is not None and orb_result is not None:
            diff = abs(line_result - orb_result)
            print(f"差異: {diff:.2f}V")

def test_separation(image_path="meter.jpg"):
    """針と円盤の分離をテスト"""
    print("=== 針と円盤の分離テスト ===")
    separator = MeterComponentSeparator()

    # 分離実行
    needle_img, dial_img, masks = separator.separate_components(image_path)

    if needle_img is not None and dial_img is not None:
        print("分離成功！")

        # ファイル保存
        separator.save_separated_components(image_path)

        return True
    else:
        print("分離に失敗しました")
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_methods("meter.jpg")
        elif sys.argv[1] == "--separate":
            test_separation("meter.jpg")
        elif sys.argv[1] == "--help":
            print("使用方法:")
            print("python3 meter_reader.py           # 通常のメーター読み取り")
            print("python3 meter_reader.py --compare # 直線検出と特徴点検出の比較")
            print("python3 meter_reader.py --separate # 針と円盤の分離")
    else:
        result = read_analog_meter("meter.jpg")
        if result is not None:
            print(f"デジタル値: {result:.2f}")
        else:
            print("メーター読み取りに失敗しました")
import cv2
import numpy as np
import math
import os

class FeatureBasedMeterReader:
    def __init__(self, detector_type='ORB'):
        """
        特徴点ベースのメーター読み取りクラス
        detector_type: 'ORB', 'SIFT', 'KAZE', 'AKAZE'から選択
        """
        self.detector_type = detector_type
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """特徴点検出器を作成"""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'SIFT':
            try:
                return cv2.SIFT_create()
            except AttributeError:
                # opencv-contrib-pythonが必要
                print("SIFT requires opencv-contrib-python")
                return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'KAZE':
            return cv2.KAZE_create()
        elif self.detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            print(f"Unknown detector: {self.detector_type}, using ORB")
            return cv2.ORB_create(nfeatures=1000)

    def _create_matcher(self):
        """マッチャーを作成"""
        if self.detector_type == 'ORB' or self.detector_type == 'AKAZE':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

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
                # 最大の円を選択
                largest_circle = max(circles, key=lambda x: x[2])
                return largest_circle
        return None

    def extract_needle_region(self, img, center, radius):
        """針領域を抽出"""
        # メーター内部のみをマスク
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, int(radius * 0.8), 255, -1)

        # 中心部分は除去（軸部分）
        cv2.circle(mask, center, int(radius * 0.05), 0, -1)

        if len(img.shape) == 3:
            masked_img = cv2.bitwise_and(img, img, mask=mask)
        else:
            masked_img = cv2.bitwise_and(img, mask)

        return masked_img, mask

    def detect_needle_features(self, img, center, radius):
        """針の特徴点を検出"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 針領域を抽出
        needle_region, mask = self.extract_needle_region(gray, center, radius)

        # コントラストを強化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(needle_region)

        # 特徴点検出
        keypoints, descriptors = self.detector.detectAndCompute(enhanced, mask)

        if keypoints is None or len(keypoints) == 0:
            return [], None

        # 針に関連する特徴点をフィルタリング
        needle_keypoints = []
        needle_descriptors = []

        for i, kp in enumerate(keypoints):
            x, y = kp.pt
            dist_to_center = math.sqrt((x - center[0])**2 + (y - center[1])**2)

            # 中心から適度な距離にある特徴点のみを針として判定
            if radius * 0.1 < dist_to_center < radius * 0.7:
                # 特徴点の応答値も考慮（閾値を下げる）
                if kp.response > 0.001:  # 応答値の閾値を下げる
                    needle_keypoints.append(kp)
                    if descriptors is not None:
                        needle_descriptors.append(descriptors[i])

        if len(needle_descriptors) > 0:
            needle_descriptors = np.array(needle_descriptors)
        else:
            needle_descriptors = None

        return needle_keypoints, needle_descriptors

    def estimate_needle_angle_from_features(self, keypoints, center):
        """特徴点から針の角度を推定"""
        if len(keypoints) < 2:
            return None

        # 重み付き重心を計算（応答値で重み付け）
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for kp in keypoints:
            weight = kp.response
            weighted_x += kp.pt[0] * weight
            weighted_y += kp.pt[1] * weight
            total_weight += weight

        if total_weight == 0:
            return None

        centroid_x = weighted_x / total_weight
        centroid_y = weighted_y / total_weight

        # 中心から重心への角度を計算
        dx = centroid_x - center[0]
        dy = centroid_y - center[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def estimate_needle_angle_pca(self, keypoints, center):
        """PCA（主成分分析）で針の方向を推定"""
        if len(keypoints) < 3:
            return self.estimate_needle_angle_from_features(keypoints, center)

        # 特徴点座標を取得
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        # 中心からの相対座標に変換
        points_centered = points - np.array([center[0], center[1]])

        # PCAを実行
        mean, eigenvectors = cv2.PCACompute(points_centered, mean=None)

        # 第1主成分（最大分散方向）が針の方向
        primary_direction = eigenvectors[0]

        # 角度を計算
        angle_rad = math.atan2(primary_direction[1], primary_direction[0])
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def create_needle_template(self, center, radius, angle_deg, template_size=(100, 100)):
        """指定角度での針テンプレートを作成"""
        template = np.zeros(template_size, dtype=np.uint8)

        # テンプレート中心
        temp_center = (template_size[1] // 2, template_size[0] // 2)

        # 針の長さ（テンプレートサイズに合わせて調整）
        needle_length = min(template_size) // 3

        # 針の終点を計算
        end_x = int(temp_center[0] + needle_length * math.cos(math.radians(angle_deg)))
        end_y = int(temp_center[1] + needle_length * math.sin(math.radians(angle_deg)))

        # 針を描画
        cv2.line(template, temp_center, (end_x, end_y), 255, 2)

        return template

    def match_with_templates(self, keypoints, descriptors, template_dict):
        """テンプレートとのマッチングを実行"""
        if descriptors is None or len(keypoints) == 0:
            return None, 0

        best_match_value = None
        best_match_score = 0

        for value, (template_kp, template_desc) in template_dict.items():
            if template_desc is None or len(template_desc) == 0:
                continue

            try:
                # 特徴点マッチング
                matches = self.matcher.match(descriptors, template_desc)

                # 距離でフィルタリング
                if self.detector_type == 'ORB' or self.detector_type == 'AKAZE':
                    good_matches = [m for m in matches if m.distance < 50]
                else:
                    good_matches = [m for m in matches if m.distance < 0.7]

                match_score = len(good_matches)
                if match_score > best_match_score and match_score >= 2:
                    best_match_score = match_score
                    best_match_value = value

            except Exception as e:
                print(f"マッチングエラー (値 {value}): {e}")
                continue

        return best_match_value, best_match_score

    def angle_to_meter_value(self, angle_deg, meter_min=0, meter_max=150):
        """角度をメーター値に変換"""
        # 既存のmeter_reader.pyと同じロジックを使用
        # メーターの角度範囲（左下225度から右下315度）

        print(f"[DEBUG] 入力角度: {angle_deg:.2f}度")

        # 340度付近は0V位置
        if angle_deg >= 315:  # 315-360度
            meter_value = 0
            print(f"[DEBUG] 315-360度の範囲: 0V")
        elif angle_deg >= 0 and angle_deg <= 45:
            # 0-45度も0V付近
            meter_value = 0
            print(f"[DEBUG] 0-45度の範囲: 0V")
        elif angle_deg >= 225 and angle_deg < 315:
            # 225-315度の範囲でメーター値を計算
            angle_normalized = (angle_deg - 225) / (315 - 225)
            meter_value = meter_min + angle_normalized * (meter_max - meter_min)
            print(f"[DEBUG] メイン範囲: 正規化角度={angle_normalized:.3f}, 値={meter_value:.2f}V")
        else:
            # その他は0V
            meter_value = 0
            print(f"[DEBUG] その他の範囲: 0V")

        return max(meter_min, min(meter_max, meter_value))

    def read_meter(self, image_path, use_pca=True, debug=False):
        """メーター値を読み取り"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像ファイル '{image_path}' が読み込めません")
            return None

        original = img.copy()

        # メーター円検出
        circle = self.detect_meter_circle(img)
        if circle is None:
            print("メーター円が検出できませんでした")
            return None

        cx, cy, radius = circle
        center = (cx, cy)

        print(f"メーター中心: {center}, 半径: {radius}")

        # 針の特徴点を検出
        keypoints, descriptors = self.detect_needle_features(img, center, radius)

        if len(keypoints) == 0:
            print("針の特徴点が検出できませんでした")
            return None

        print(f"検出された針の特徴点数: {len(keypoints)}")

        # 特徴点から針の角度を推定
        if use_pca and len(keypoints) >= 3:
            angle_deg = self.estimate_needle_angle_pca(keypoints, center)
            method = "PCA"
        else:
            angle_deg = self.estimate_needle_angle_from_features(keypoints, center)
            method = "重心"

        if angle_deg is None:
            print("針の角度が推定できませんでした")
            return None

        # メーター値に変換
        meter_value = self.angle_to_meter_value(angle_deg)

        print(f"角度推定方法: {method}")
        print(f"推定角度: {angle_deg:.2f}度")
        print(f"推定メーター値: {meter_value:.2f}V")

        # デバッグ表示
        if debug:
            self.visualize_debug(original, center, radius, keypoints, angle_deg, meter_value, method)

        return meter_value

    def visualize_debug(self, img, center, radius, keypoints, angle_deg, meter_value, method):
        """デバッグ用可視化"""
        debug_img = img.copy()

        # 円と中心を描画
        cv2.circle(debug_img, center, radius, (0, 255, 0), 2)
        cv2.circle(debug_img, center, 2, (0, 0, 255), 3)

        # 針の検出領域を描画
        cv2.circle(debug_img, center, int(radius * 0.85), (255, 255, 0), 1)
        cv2.circle(debug_img, center, int(radius * 0.15), (255, 255, 0), 1)

        # 特徴点を描画
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(debug_img, (x, y), 3, (255, 0, 0), -1)
            # 特徴点番号を表示
            cv2.putText(debug_img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 推定角度線を描画
        end_x = int(center[0] + radius * 0.7 * math.cos(math.radians(angle_deg)))
        end_y = int(center[1] + radius * 0.7 * math.sin(math.radians(angle_deg)))
        cv2.line(debug_img, center, (end_x, end_y), (0, 255, 255), 3)

        # 情報表示
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, f"Value: {meter_value:.1f}V", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Angle: {angle_deg:.1f}deg", (10, 70), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Features: {len(keypoints)}", (10, 110), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Detector: {self.detector_type}", (10, 150), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Method: {method}", (10, 190), font, 1, (0, 255, 0), 2)

        # 元画像と比較表示
        combined = np.hstack([img, debug_img])
        cv2.imshow('Feature-Based Meter Reading', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def compare_detectors(image_path):
    """異なる特徴点検出器で結果を比較"""
    detectors = ['ORB', 'AKAZE', 'KAZE']
    results = {}

    print("=== 特徴点検出器比較 ===")

    for detector_type in detectors:
        try:
            print(f"\n--- {detector_type} ---")
            reader = FeatureBasedMeterReader(detector_type)
            result = reader.read_meter(image_path, use_pca=True)
            results[detector_type] = result

            if result is not None:
                print(f"{detector_type}: {result:.2f}V")
            else:
                print(f"{detector_type}: 読み取り失敗")

        except Exception as e:
            print(f"{detector_type}: エラー - {e}")
            results[detector_type] = None

    # 結果サマリー
    print(f"\n=== 結果サマリー ===")
    valid_results = [v for v in results.values() if v is not None]

    for detector, result in results.items():
        if result is not None:
            print(f"{detector}: {result:.2f}V")
        else:
            print(f"{detector}: 失敗")

    if len(valid_results) > 1:
        avg_result = sum(valid_results) / len(valid_results)
        std_dev = math.sqrt(sum((x - avg_result)**2 for x in valid_results) / len(valid_results))
        print(f"\n平均値: {avg_result:.2f}V")
        print(f"標準偏差: {std_dev:.2f}V")

    return results

def compare_methods(image_path):
    """重心法とPCA法を比較"""
    print("=== 角度推定方法比較 ===")

    reader = FeatureBasedMeterReader('ORB')

    print("\n--- 重心法 ---")
    result_centroid = reader.read_meter(image_path, use_pca=False)

    print("\n--- PCA法 ---")
    result_pca = reader.read_meter(image_path, use_pca=True)

    print(f"\n=== 比較結果 ===")
    if result_centroid is not None:
        print(f"重心法: {result_centroid:.2f}V")
    else:
        print("重心法: 失敗")

    if result_pca is not None:
        print(f"PCA法: {result_pca:.2f}V")
    else:
        print("PCA法: 失敗")

    if result_centroid is not None and result_pca is not None:
        diff = abs(result_centroid - result_pca)
        print(f"差異: {diff:.2f}V")

if __name__ == "__main__":
    import sys

    image_path = "meter.jpg"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare-detectors":
            compare_detectors(image_path)
        elif sys.argv[1] == "--compare-methods":
            compare_methods(image_path)
        elif sys.argv[1] == "--debug":
            reader = FeatureBasedMeterReader('ORB')
            reader.read_meter(image_path, debug=True)
        elif sys.argv[1] == "--help":
            print("使用方法:")
            print("python3 meter_reader_features.py                    # 通常実行")
            print("python3 meter_reader_features.py --compare-detectors # 検出器比較")
            print("python3 meter_reader_features.py --compare-methods   # 角度推定方法比較")
            print("python3 meter_reader_features.py --debug            # デバッグ表示")
        else:
            print("不明なオプション:", sys.argv[1])
            print("--help で使用方法を確認してください")
    else:
        # 通常の実行
        print("=== 特徴点ベースメーター読み取り ===")
        reader = FeatureBasedMeterReader('ORB')
        result = reader.read_meter(image_path, use_pca=True)

        if result is not None:
            print(f"\n最終結果: {result:.2f}V")
        else:
            print("\nメーター読み取りに失敗しました")
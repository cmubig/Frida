import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def find_arucos(img: np.ndarray, show: bool = False) -> np.ndarray:
    # the final corners will be in the order of
    # [top-left, top-right, bottom-right, bottom-left]
    final_corners = [
        'top-left',
        'top-right',
        'bottom-right',
        'bottom-left',
    ]
    aruco_ids = {0: 'bottom-left',
                 1: 'bottom-right',
                 2: 'top-right',
                 3: 'top-left'}

    # aruco_ids = {0: 'bottom-left',
    #              2: 'top-right',
    #              3: 'top-left',
    #              4: 'bottom-right'}

    # Map marker id to its corresponding corner label
    id_to_label = {v: k for k, v in aruco_ids.items()}

    # Get the indices that would sort corners as [top-left, top-right, bottom-right, bottom-left]
    label_order = [id_to_label[label] for label in final_corners]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Define the ArUco dictionary and parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters = cv.aruco.DetectorParameters()
    # Create ArucoDetector and detect markers
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    if ids is None or len(ids) != 4:
        print("Error: Could not find 4 Aruco markers.")
        return None, None
    img_markers = img.copy()


    sorted_idx = [np.where(ids.flatten() == i)[0][0] for i in label_order]
    corners = [corners[i][0] for i in sorted_idx]
    ids = ids[sorted_idx]
    ids = ids.flatten()

    # Map marker id to its furthest point (assuming each marker's corners are ordered)
    # Compute the centroid of all detected marker corners
    all_pts = np.concatenate(corners, axis=0)
    centroid = np.mean(all_pts, axis=0)
    ordered_corners = [None] * 4
    for idx, marker_corners in enumerate(corners):
        # For each marker, find the corner furthest from the centroid
        dists = np.linalg.norm(marker_corners - centroid, axis=1)
        furthest_idx = np.argmax(dists)
        ordered_corners[idx] = marker_corners[furthest_idx]

    if show:
        # Draw detected markers and their IDs

        for idx, pt in enumerate(ordered_corners):
            pt_int = tuple(np.round(pt).astype(int))
            cv.circle(img_markers, pt_int, radius=10, color=(0, 0, 255), thickness=-1)
            cv.putText(img_markers, f"{aruco_ids[ids[idx]]}", pt_int,
                    cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)

        plt.imshow(cv.cvtColor(img_markers, cv.COLOR_BGR2RGB))
        plt.title("Aruco marker points")
        plt.show()
        plt.close()

    ordered_corners = np.array(ordered_corners, dtype=np.int64)

    return ordered_corners


if __name__ == "__main__":
    img = cv.imread('images/test.png')
    corners = find_arucos(img, True)

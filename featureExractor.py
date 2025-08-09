import numpy as np
import cv2
from sklearn.cluster import MeanShift

def ellipse_roundness(binary_mask):
    """
    Compute roundness of a binary region based on how many 'true' pixels are inside the fitted ellipse.
    
    Parameters:
        binary_mask (numpy.ndarray): Binary image where 1 = lesion, 0 = background.

    Returns:
        float: Roundness score (1 = perfect circle, closer to 0 for irregular shapes).
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0.0  # No contours found
    
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return 0.0  # Not enough points for ellipse fitting
    
    # Fit an ellipse to the largest contour
    ellipse = cv2.fitEllipse(largest_contour)
    center, axes, angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # Semi-major and semi-minor axes

    # Create an empty mask for the ellipse
    ellipse_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Draw the ellipse on the empty mask
    cv2.ellipse(ellipse_mask, (int(center[0]), int(center[1])), (int(a), int(b)), angle, 0, 360, 255, thickness=-1)

    # Count the number of 'true' pixels in the original binary mask and in the ellipse mask
    total_true_pixels = np.sum(binary_mask == 1)
    true_pixels_inside_ellipse = np.sum((binary_mask == 1) & (ellipse_mask == 255))

    # If there are no 'true' pixels in the binary mask, return 0
    if total_true_pixels == 0:
        return 0.0
    
    # Compute the ratio of 'true' pixels inside the ellipse to the total number of 'true' pixels
    roundness_score = true_pixels_inside_ellipse / total_true_pixels
    return roundness_score


def normalize_features(features):
    """
    Normalize feature values to be between 0 and 1.
    """
    min_vals = np.min(features)
    max_vals = np.max(features)
    
    if max_vals - min_vals == 0:
        return np.zeros_like(features)  # Avoid division by zero
    
    return (features - min_vals) / (max_vals - min_vals)


def describe_skin_lesion(rgb_image, binary_mask):
    """
    Extract a 7D feature vector for a skin lesion.
    
    Parameters:
        rgb_image (numpy.ndarray): Input RGB image (H, W, 3).
        binary_mask (numpy.ndarray): Binary image (H, W), 1 = lesion, 0 = background.

    Returns:
        numpy.ndarray: Normalized 1x7 feature vector [Mean L, Mean A, Mean B, Std L, Std A, Std B, Roundness].
    """
    # Convert to LAB color space
    rgb_image = cv2.cvtColor(rgb_image, cv2.cv.CV_BGR2RGB)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    # Extract lesion pixels
    lesion_pixels = lab_image[binary_mask == 1]
    
    if lesion_pixels.size == 0:  # If no lesion pixels, return zero vector
        return np.zeros(7)
    
    # Compute mean and standard deviation for L, A, B channels
    mean_L, mean_A, mean_B = np.mean(lesion_pixels, axis=0)
    std_L, std_A, std_B = np.std(lesion_pixels, axis=0)
    
    # Normalize the mean and standard deviation values (0 to 255)
    mean_L = mean_L / 255
    std_L = std_L / 255
    mean_A = mean_A / 255
    mean_B = mean_B / 255
    std_A = std_A / 255
    std_B = std_B / 255
    
    # Compute roundness score
    roundness = ellipse_roundness(binary_mask)
    
    # Create feature vector
    feature_vector = np.array([mean_L, mean_A, mean_B, std_L, std_A, std_B, roundness])
    
    # Return the normalized feature vector
    return feature_vector


def segment_image(image, bandwidth=20):
    # Load image
    image = cv2.cvtColor(np.array(image), cv2.cv.CV_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))  # Separates into RGB channels
    
    # Mean Shift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(pixels)
    
    # Get the labels and cluster centers
    labels = meanshift.labels_
    cluster_centers = meanshift.cluster_centers_
    print(cluster_centers)
    
    # Reshape the labels to the original image dimensions
    segmented_image = cluster_centers[labels].reshape(image.shape)  # Assign each pixel to its cluster center color
    gray_segmented = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    threshold_value = 190  # Lower values make more areas white, adjust as needed
    _, binary_image = cv2.threshold(gray_segmented, threshold_value, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_not(binary_image)
    return binary_image


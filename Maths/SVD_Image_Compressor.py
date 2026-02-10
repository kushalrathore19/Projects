import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

image = data.camera() 

# image = io.imread('your_image_path.jpg', as_gray=True)

# normalising it to [0, 1]
image_matrix = image / 255.0

print(f"Original Image Shape: {image_matrix.shape}")

# Perform SVD (singular value decomposition)
# full_matrices=False makes it efficient (Economy SVD)
# U is Left Singular Vectors (The "Output" patterns)
# S is Singular Values (The "Strengths" - 1D array)
# Vt is Right Singular Vectors Transposed (The "Input" patterns)
U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"S shape: {S.shape}")
print(f"Vt shape: {Vt.shape}")

def reconstruct_svd(U, S, Vt, k):
    """
    Reconstructs the image using only the top k singular values. for effeicent, we only keep the top k components of U, S, and Vt.
    """
    # 1. Slice the matrices to keep only k components
    U_k = U[:, :k] 
    S_k = np.diag(S[:k]) # top k to a diagonal matrix
    Vt_k = Vt[:k, :]
    
    # 2. Rebuild the image: U * S * Vt
    reconstructed_image = np.dot(U_k, np.dot(S_k, Vt_k))
    
    return reconstructed_image

# Let's test it with k=50 (Using top 50 patterns out of 512)
k_test = 50
img_compressed = reconstruct_svd(U, S, Vt, k_test)

# Plotting the original and compressed images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_matrix, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed (k={k_test})")
plt.imshow(img_compressed, cmap='gray')
plt.axis('off')

plt.show()

# List of K values to test 
k_values = [5, 10, 20, 50, 100, 200, 400]
errors = []

plt.figure(figsize=(15, 8))

# Loop through k values
for i, k in enumerate(k_values):
    # Reconstruct
    img_k = reconstruct_svd(U, S, Vt, k) # applyig the function to get the compressed image for each k
    
    # Calculate Error (Frobenius Norm)
    diff = image_matrix - img_k
    error = np.linalg.norm(diff) / np.linalg.norm(image_matrix) * 100
    errors.append(error)
    
    # Plot the images for the first 6 k values (we can use more subplots if we want to show all)
    if i < 6:
        plt.subplot(2, 3, i+1)
        plt.title(f"k={k}\nError: {error:.2f}%")
        plt.imshow(img_k, cmap='gray')
        plt.axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

# Plot the Singular Values (The "Scree Plot")
# This shows how fast the "energy" drops off
plt.subplot(1, 2, 1)
plt.semilogy(S) # Log scale to see the drop better
plt.title("Singular Values (Energy Profile)")
plt.xlabel("Index")
plt.ylabel("Singular Value (Log Scale)")
plt.grid(True)

# Plot Reconstruction Error vs k to see how error decreases as we keep more components
plt.subplot(1, 2, 2)
plt.plot(k_values, errors, marker='o', color='red')
plt.title("Reconstruction Error vs. k")
plt.xlabel("Rank (k)")
plt.ylabel("Error (%)")
plt.grid(True)

plt.show()


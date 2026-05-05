import cv2
import matplotlib.pyplot as plt

img = cv2.imread("gambar.jpg")

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

print("Gambar berhasil dibaca!")

# Original
cv2.imshow("Original", img)
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Histogram
plt.hist(gray.ravel(), bins=256, range=[0,256])
plt.title("Histogram")
plt.show()

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#FLIP
flip = cv2.flip(img, 1)

cv2.imshow("Flip", flip)
cv2.waitKey(0)

#ROTATE
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Rotate", rotate)
cv2.waitKey(0)

#CROP
h, w = img.shape[:2]
crop = img[h//4:3*h//4, w//4:3*w//4]

cv2.imshow("Crop", crop)
cv2.waitKey(0)

#PENUTUP
cv2.destroyAllWindows()

print("\nAnalisis:")
print("- Grayscale menyederhanakan citra dari 3 channel menjadi 1 channel")
print("- Histogram menunjukkan distribusi intensitas pixel")
print("- Threshold digunakan untuk memisahkan objek dan background")
print("- Flip mengubah orientasi horizontal")
print("- Rotate memutar gambar 90 derajat")
print("- Crop mengambil sebagian area dari citra")
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

std::uint32_t readBigEndianInt(std::ifstream &file) {
  unsigned char bytes[4];
  file.read(reinterpret_cast<char *>(bytes), 4);

  if (!file) {
    throw std::runtime_error("Could not read IDX header");
  }

  return (static_cast<std::uint32_t>(bytes[0]) << 24) |
         (static_cast<std::uint32_t>(bytes[1]) << 16) |
         (static_cast<std::uint32_t>(bytes[2]) << 8) |
         static_cast<std::uint32_t>(bytes[3]);
}

std::vector<double> loadImageVector(std::size_t index) {
  std::ifstream images("data/train-images.idx3-ubyte", std::ios::binary);

  if (!images) {
    throw std::runtime_error("Could not open data/train-images.idx3-ubyte");
  }

  // Read the IDX image header.
  const std::uint32_t magic = readBigEndianInt(images);
  const std::uint32_t imageCount = readBigEndianInt(images);
  const std::uint32_t rows = readBigEndianInt(images);
  const std::uint32_t cols = readBigEndianInt(images);

  if (magic != 2051 || rows != 28 || cols != 28) {
    throw std::runtime_error("Invalid MNIST image file");
  }

  if (index >= imageCount) {
    throw std::out_of_range("Image index is outside the dataset");
  }

  // Each image has 784 bytes. The 16-byte header comes first.
  const std::streamoff offset =
      16 + static_cast<std::streamoff>(index) * rows * cols;

  images.seekg(offset, std::ios::beg);
  if (!images) {
    throw std::runtime_error("Could not seek to the selected image");
  }

  // Read the selected image's raw pixel bytes.
  std::vector<unsigned char> pixels(rows * cols);
  images.read(reinterpret_cast<char *>(pixels.data()),
              static_cast<std::streamsize>(pixels.size()));

  if (!images) {
    throw std::runtime_error("Could not read image pixels");
  }

  // Convert the bytes (0–255) to network inputs (0.0–1.0).
  std::vector<double> imageVector;
  imageVector.reserve(pixels.size());

  for (unsigned char pixel : pixels) {
    imageVector.push_back(static_cast<double>(pixel) / 255.0);
  }

  return imageVector;
}

int main() {
  long long enteredIndex;

  std::cout << "Enter an image index (0-59999): ";
  if (!(std::cin >> enteredIndex) || enteredIndex < 0) {
    std::cerr << "Enter a valid non-negative index.\n";
    return 1;
  }

  try {
    // This is the vector output from loadImageVector.
    std::vector<double> input =
        loadImageVector(static_cast<std::size_t>(enteredIndex));

    std::cout << "Vector size: " << input.size() << '\n';
    std::cout << "Vector values:\n";

    for (double pixel : input) {
      std::cout << pixel << ' ';
    }
    std::cout << '\n';

    // Pass `input` to your neural network here.
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}

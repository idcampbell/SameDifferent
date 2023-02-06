# Generate an image dataset with 8 shapes and 8 colors with user specified colors.
python generate_data.py \
  image \
  --name image_8s8c_categorical \
  --nShapes 8 \
  --nColors 8 \
  --path data/sprites.npy \
  --colors blue fuchsia grey lime maroon teal yellow orange \
  --splits 2 3 4 5 6 7 8

# Generate an image dataset with 16 shapes and 16 colors from the RGB colormap.
python generate_data.py \
  image \
  --name image_16s16c_continuous \
  --nShapes 16 \
  --nColors 16 \
  --path data/sprites.npy \
  --cmap brg \
  --splits 2 3 4 5 6 7 8

# Generate a one-hot dataset with 16 shapes and 16 colors.
python generate_data.py \
  one-hot \
  --name onehot_16s16c \
  --nShapes 16 \
  --nColors 16 \
  --splits 2 3 4 5 6 7 8




# Train an image model on two splits for the categorical color data.
python train.py \
  image \
  --name image_16s16c_categorical \
  --splits 8 \
  --device cpu \
  --logInterval 10 \
  --nEpochs 1000

# Train an image model on two splits for the continuous colors.
python train.py \
  image \
  --name image_16s16c_continuous \
  --splits 4 7 \
  --device cpu \
  --logInterval 100 \
  --nEpochs 1000

# Train a linear model on some splits.
python train.py \
  one-hot \
  --name onehot_16s16c \
  --splits 2 3 4 5 6 7 8 \
  --device cpu \
  --logInterval 100
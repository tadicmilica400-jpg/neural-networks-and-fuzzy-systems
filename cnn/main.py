import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ucitavanje CSV
if not os.path.isfile('Patterns.csv') or not os.path.isdir('Patterns'):
    raise SystemExit('Missing Patterns.csv or Patterns/. See README.md for dataset requirements.')
tf.keras.utils.set_random_seed(123)
df = pd.read_csv("Patterns.csv")
if not {'ClassName', 'Path'}.issubset(df.columns) or df[['ClassName', 'Path']].isna().any().any():
    raise SystemExit('Patterns.csv must contain non-missing ClassName and Path values.')
if df['ClassName'].nunique() != 2:
    raise SystemExit('Expected exactly two classes.')
if df['Path'].map(lambda x: os.path.basename(str(x))).duplicated().any():
    raise SystemExit('Duplicate image filenames found; resolve duplicate samples before splitting.')
if any(str(x) in {'.', '..'} or '/' in str(x) or '\\' in str(x) for x in df['ClassName']):
    raise SystemExit('Class names must be plain directory names.')
missing = [str(x) for x in df['Path'] if not os.path.isfile(os.path.join('Patterns', os.path.basename(str(x))))]
if missing:
    raise SystemExit(f'{len(missing)} referenced image files are missing from Patterns/.')
if os.path.isdir('data') and any(os.scandir('data')):
    raise SystemExit('data/ is non-empty. Move the old generated folder aside before running to avoid stale samples.')

print(df.head())
#koliko ima koje klase
print(df["ClassName"].value_counts())

#histogram raspodele odbiraka po klasama
plt.figure(figsize=(6,4))
sns.countplot(x="ClassName", data=df)
plt.title("Broj uzoraka po klasama")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_folder = "data"
images_folder = "Patterns"

#pravljenje foldera po klasama
for index, row in df.iterrows():
    class_name = row["ClassName"]
    image_name = os.path.basename(row["Path"])

    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)#ako folder ne postoji kreira ga

    src = os.path.join(images_folder, image_name)
    dst = os.path.join(class_folder, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)


train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2, #20% - validacija, 80% - trening
    subset="training", #vraca skup za trening
    seed=123,
    image_size=(128,128), #sve slike se skaliraju
    batch_size=32 #posle 32 slike azurira tezine-back propagation
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation", #vraca skup za validaciju
    seed=123,
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names
print("Klase:", class_names)

#normalizacija, slike su u opsegu 0-255 -> 0-1
normalization_layer = layers.Rescaling(1./255)

#normalizujemo sve ulaze, ne i izlaze
#.map() primeni tu funkciju na sve elemente dataseta
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(), #pretvara 3d u 1d vektor (priprema za dense sloj)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), #50% neurona se gasi nasumicno
    layers.Dense(len(class_names), activation='softmax')
])
print("Summary: ")
model.summary()

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='sparse_categorical_crossentropy', #sparse jer je 0/1 a ne 01/10
    metrics=['accuracy']
)
# compile() tells the model three things:
# How to update weights → optimizer
# How to measure error → loss
# What to report during training → metrics

history = model.fit( #pokrece proces ucenja
    train_ds,
    validation_data=val_ds,
    epochs=10
)

#crtanje grafika greske
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Greska tokom epoha')
plt.xlabel('Epoha')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

y_true = [] #stvarne klase
y_pred = [] #predikcije

for images, labels in val_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1)) #uzima indeks najvece vrednost po svakom redu (axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predikcija")
plt.ylabel("Stvarna klasa")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

#izdvajanje tacnih i pogresnih primera
correct_images = []
correct_labels = []
correct_preds = []

wrong_images = []
wrong_labels = []
wrong_preds = []

for images, labels in val_ds:
    predictions = model.predict(images)
    preds = np.argmax(predictions, axis=1)

    for i in range(len(labels)):
        if labels[i] == preds[i]:
            correct_images.append(images[i])
            correct_labels.append(labels[i])
            correct_preds.append(preds[i])
        else:
            wrong_images.append(images[i])
            wrong_labels.append(labels[i])
            wrong_preds.append(preds[i])

plt.figure(figsize=(10,5))

for i in range(min(5, len(correct_images))):
    plt.subplot(1,5,i+1)
    plt.imshow(correct_images[i])
    plt.title(f"True: {class_names[correct_labels[i]]}\nPred: {class_names[correct_preds[i]]}")
    plt.axis("off")

plt.suptitle("Primeri dobro klasifikovanih slika")
plt.show()

plt.figure(figsize=(10,5))

for i in range(min(5, len(wrong_images))):
    plt.subplot(1,5,i+1)
    plt.imshow(wrong_images[i])
    plt.title(f"True: {class_names[wrong_labels[i]]}\nPred: {class_names[wrong_preds[i]]}")
    plt.axis("off")

plt.suptitle("Primeri pogrešno klasifikovanih slika")
plt.show()

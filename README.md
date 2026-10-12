 Photo Finder — Wedding Photo Face Search

FaceFinder is a Python desktop application that helps you find photos of a particular person across a large photo album using AI-powered face recognition.

It is designed for wedding albums and other large photo collections. Instead of manually checking thousands of pictures, you provide one or more reference photos, let FaceFinder index the album, and review the photos that may contain the same person.

> **Note:** Face matches are predictions, not guaranteed identity verification. Always review the results before sharing or using exported photos.

## ✨ Features

- **AI-powered face recognition** using InsightFace.
- **Multiple reference photos** to improve matching across different angles, expressions, and lighting conditions.
- **Webcam capture** to use a live camera image as a reference.
- **Group-photo face selection** when a reference image contains multiple people.
- **Persistent photo indexing** using SQLite to avoid reprocessing unchanged images on every search.
- **Automatic index updates** when photos are added or modified.
- **Saved folder locations** so source and destination folders persist between sessions.
- **HEIC and HEIF support**, in addition to common image formats.
- **Thumbnail gallery** to review potential matches visually.
- **Similarity scores and confidence filters** to help prioritize results.
- **Sorting options** by match score, estimated face quality, date, or filename.
- **Named person profiles** to save and reload reference photos.
- **Exact duplicate detection** using file hashes.
- **Selective export** to copy chosen matching photos into a destination folder.
- **CSV export manifest** containing source paths, exported paths, similarity scores, and timestamps.
- **Background processing** to keep the desktop interface responsive during indexing and searching.

## 🖥️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Tkinter | Desktop graphical interface |
| InsightFace | Face detection and recognition embeddings |
| ONNX Runtime | AI model inference |
| OpenCV | Webcam capture and image processing |
| Pillow | Image loading, orientation correction, and previews |
| pillow-heif | HEIC/HEIF image support |
| NumPy | Embedding normalization and similarity calculations |
| SQLite | Persistent photo index and saved person profiles |

## 📋 Requirements

- Windows 10 or Windows 11 recommended
- Python 3.10 or compatible Python version
- Webcam, optional
- Internet connection for the initial model download, if the model is not already cached
- Enough disk space for the AI model and local face index

The application uses CPU inference by default for compatibility.

## 🚀 Installation

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/FaceFinder.git
cd FaceFinder
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows Command Prompt:

```bat
.venv\Scripts\activate
```

In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the VS Code terminal with Command Prompt or run Python through the virtual environment directly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install insightface onnxruntime opencv-python pillow pillow-heif numpy
```

**Note:** InsightFace installation can vary by Python version and Windows environment. If installation fails, check the error message and the installation requirements for the relevant package.

### 4. Run FaceFinder

```bash
python face_finder.py
```

On the first launch, the InsightFace model may take additional time to download and initialize.

## 📖 How to Use

### Step 1: Set your folders

Click **Change Folder Locations** and select:

- **Source folder:** The folder containing your wedding album.
- **Destination folder:** The folder where selected matches should be copied.

FaceFinder saves these locations for subsequent sessions.

### Step 2: Add reference photos

Choose **Add reference photo(s)** and select one or more clear pictures of the person you want to find.

If a picture contains multiple faces, select the correct face when prompted.

You can also use **Capture webcam photo** to capture a reference with your webcam.

**Tips for better references:**
- Use clear, well-lit images.
- Include different angles and expressions.
- Prefer photos where the person's face is reasonably large.
- Avoid using only blurry or heavily obstructed faces.
- Make sure all reference photos belong to the same person.

### Step 3: Save a person profile (optional)

Enter a name and click **Save profile** to save the selected reference-photo paths. Later, choose the profile and click **Load profile**.

Keep the reference images at their original paths so the saved profile can load them in future sessions.

### Step 4: Search the album

Click **Find Person in Album**.

FaceFinder scans supported images, detects faces, stores their embeddings in its local index, and compares them with your reference photos.

The first scan may take significantly longer than later searches. Subsequent searches can reuse cached results for unchanged files.

### Step 5: Review the results

The results gallery shows candidate photos with similarity scores and additional information.

You can:

- Sort by match score, estimated face quality, date, or filename.
- Filter results by similarity score.
- Click a thumbnail to preview the original image.
- Select or deselect photos for export.
- Inspect exact duplicate-file groups.

### Step 6: Export selected photos

Select the photos you want and click **Export checked matches**.

FaceFinder copies the selected files to a person-named subfolder within your destination folder and creates a CSV manifest of the export.

The original photos are not intentionally moved or deleted.

## 🎯 Accuracy and Similarity Threshold

FaceFinder uses normalized face embeddings and cosine similarity to rank candidate matches.

The default threshold is `0.40`.

- **Higher threshold:** Fewer candidate photos, potentially fewer false positives, but genuine matches may be missed.
- **Lower threshold:** More candidate photos, potentially more genuine matches, but also more false positives.

The appropriate threshold depends on the image quality, camera angles, lighting, age differences, and characteristics of the photo collection.

The score is a similarity measure, **not a calibrated probability that two faces belong to the same person**. A score such as `0.60` does not mean a 60% chance of identity.

For important results, inspect the original images rather than relying only on the score or thumbnail.

## 🗂️ Local Data Files

The application creates or uses local files alongside the Python script:

| File or folder | Purpose |
|---|---|
| `facefinder_config.json` | Saved source folder, destination folder, and threshold |
| `facefinder_index.sqlite3` | Cached photo information, face embeddings, and named profiles |
| `facefinder_references/` | Saved webcam reference images |

Keep these files if you want to retain your local settings and index between runs.

If you move or rename the project folder, verify your saved paths. You can use **Change Folder Locations** to select the correct folders again.

## 🖼️ Supported Image Formats

- JPG / JPEG
- PNG
- WebP
- BMP
- TIFF
- HEIC / HEIF

Actual readability depends on the installed image libraries and the individual file.

## ⚡ Performance

For a large album of several thousand images:

- The first indexing run may take a while because every image needs to be examined.
- Later searches can be faster because unchanged images are cached.
- Processing time depends on image resolution, the number of detected faces, CPU performance, and storage speed.
- Keeping the SQLite database between sessions avoids unnecessary re-indexing.

If you add or edit photos, the application checks file metadata and refreshes the relevant index entries.

## 🔧 Troubleshooting

### InsightFace or ONNX Runtime fails to install

Confirm your Python version:

```bash
python --version
```

Then update pip and retry the dependency installation. Package compatibility can vary across Python versions and Windows environments.

### No face detected in a reference photo

Try a clearer image with better lighting and a more visible face. A very small, turned-away, blurry, or obstructed face may not be detected.

### Too many unrelated results

Increase the similarity threshold gradually and use multiple clear reference photos of the same person.

### Genuine matches are missing

Try lowering the threshold slightly and adding reference photos showing different angles or lighting conditions.

### HEIC files cannot be opened

Confirm that `pillow-heif` is installed:

```bash
python -m pip install pillow-heif
```

### Webcam does not open

Check that Windows camera permissions are enabled and that another application is not exclusively using the camera.

### A search seems slow

The initial scan is the most expensive. Keep the SQLite index file between sessions so unchanged photos can be reused.

## 🔐 Privacy

FaceFinder processes the selected photo collection locally in the application. The local index contains face embeddings, which are sensitive biometric data, as well as image paths and metadata.

- Do not publish your photo album, face embeddings, database, or private reference images to a public repository.
- Only scan images you have permission to use.
- Protect local index and configuration files as you would other private data.
- Review applicable privacy requirements before using the application on other people's photos.

## 🛣️ Possible Future Improvements

- GPU acceleration where supported
- More advanced face-quality and match-ranking methods
- Near-duplicate and burst-photo detection
- A more advanced contact-sheet review interface
- Better indexing for network drives and very large collections
- Search history and resumable export jobs
- Configurable model and detection settings

## ⚠️ Limitations

- Face recognition can produce false positives and false negatives.
- Performance and accuracy vary with image quality and face visibility.
- Similarity thresholds require practical tuning for each collection.
- The index is local to the application and is not a cloud synchronization system.
- The application has not been validated for every camera, image format, hardware configuration, or photo collection.

## 📄 License

Choose a license before distributing this project publicly. If you choose the MIT License, add a `LICENSE` file containing the appropriate MIT License text.

Review the licenses and usage terms of InsightFace, its pretrained models, and all other dependencies before distributing the application or model files.

---

**Built with Python to make searching large wedding photo collections easier.**

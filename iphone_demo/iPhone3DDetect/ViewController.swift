import UIKit
import ARKit
import SceneKit
import ImageIO
import PhotosUI

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate, PHPickerViewControllerDelegate {

    private var sceneView: ARSCNView!
    private var bboxRenderer: BBoxRenderer!
    private var isDetecting = false

    // HUD
    private var statusLabel: UILabel!
    private var captureButton: UIButton!
    private var activityIndicator: UIActivityIndicatorView!
    private var settingsButton: UIButton!
    private var promptBar: UIView!
    private var chipsScrollView: UIScrollView!
    private var chipsStack: UIStackView!
    private var modeSegment: UISegmentedControl!
    private var captionLabel: UILabel!

    // Class management (persisted)
    private var savedClasses: [String] = []
    private var activeClasses: Set<String> = []
    private var captureSegment: UISegmentedControl!
    private var sendDepth = true
    private var alignmentPostProcess = true
    private var unitMode = 0  // 0=meters, 1=feet, 2=cm
    private var useLocalModel = false  // Toggle for local ONNX inference
    private var captureMode = 0  // 0=Single, 1=Multi-View, 2=Video
    private var clearButton: UIButton!
    private var videoFrames: [CaptureData] = []
    private var videoTimer: Timer?
    private let videoFrameCount = 15
    private let videoFrameInterval: TimeInterval = 0.2
    private var multiViewFrames: [CaptureData] = []
    private var multiViewDetectButton: UIButton!

    // Detection method: 0=Classes, 1=2D Box
    private var detectionMethod = 0
    private var detectMethodSegment: UISegmentedControl!
    private var panGesture: UIPanGestureRecognizer!

    // 2D box drag — multiple boxes
    private var dragStartPoint: CGPoint?
    private var drawnBoxViews: [UIView] = []
    private var drawnBoxCoords: [[Float]] = []
    private var box2DCaptureData: CaptureData?

    // Box selection & deletion
    private var selectedBox2DIndex: Int?
    private var deleteButton: UIButton!

    // Input mode: 0=Camera, 1=Upload
    private var inputMode = 0
    private var inputModeSegment: UISegmentedControl!
    private var uploadImageView: UIImageView!
    private var uploadedImage: UIImage?
    private var uploadPngData: Data?  // For upload mode box2D (no CaptureData needed)
    private var saveButton: UIButton!
    private var pickPhotoButton: UIButton!
    private var changePhotoButton: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupSceneView()
        setupHUD()
        loadClasses()
        rebuildChips()
        bboxRenderer = BBoxRenderer(scene: sceneView.scene)
        DetectionService.shared.warmUp()
    }

    private var isLiDARAvailable = false

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if inputMode == 0 {
            startARSession()
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }

    private func startARSession() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
        isLiDARAvailable = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        if isLiDARAvailable {
            config.frameSemantics = .sceneDepth
            print("[AR] LiDAR depth enabled")
        } else {
            print("[AR] No LiDAR - monocular fallback")
        }
        sceneView.session.run(config)
    }

    // MARK: - Scene View

    private func setupSceneView() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.automaticallyUpdatesLighting = true
        view.addSubview(sceneView)

        uploadImageView = UIImageView(frame: view.bounds)
        uploadImageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        uploadImageView.contentMode = .scaleAspectFit
        uploadImageView.backgroundColor = .black
        uploadImageView.isHidden = true
        uploadImageView.isUserInteractionEnabled = true
        view.addSubview(uploadImageView)

        panGesture = UIPanGestureRecognizer(target: self, action: #selector(handleBoxDrag(_:)))
        panGesture.maximumNumberOfTouches = 1
        panGesture.isEnabled = false
        view.addGestureRecognizer(panGesture)

        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        view.addGestureRecognizer(tapGesture)
    }

    // MARK: - HUD

    private func setupHUD() {
        statusLabel = UILabel()
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.textColor = .white
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        statusLabel.font = .systemFont(ofSize: 14, weight: .medium)
        statusLabel.textAlignment = .center
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        statusLabel.text = " Ready - Tap Detect "
        view.addSubview(statusLabel)

        captureButton = UIButton(type: .system)
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.setTitle("  Detect  ", for: .normal)
        captureButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .bold)
        captureButton.setTitleColor(.white, for: .normal)
        captureButton.backgroundColor = UIColor.systemBlue
        captureButton.layer.cornerRadius = 25
        captureButton.addTarget(self, action: #selector(captureAndDetect), for: .touchUpInside)
        view.addSubview(captureButton)

        multiViewDetectButton = UIButton(type: .system)
        multiViewDetectButton.translatesAutoresizingMaskIntoConstraints = false
        multiViewDetectButton.setTitle("  Detect  ", for: .normal)
        multiViewDetectButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .bold)
        multiViewDetectButton.setTitleColor(.white, for: .normal)
        multiViewDetectButton.backgroundColor = UIColor.systemBlue
        multiViewDetectButton.layer.cornerRadius = 25
        multiViewDetectButton.addTarget(self, action: #selector(sendMultiViewDetection), for: .touchUpInside)
        multiViewDetectButton.isHidden = true
        view.addSubview(multiViewDetectButton)

        activityIndicator = UIActivityIndicatorView(style: .medium)
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.color = .white
        activityIndicator.hidesWhenStopped = true
        view.addSubview(activityIndicator)

        settingsButton = UIButton(type: .system)
        settingsButton.translatesAutoresizingMaskIntoConstraints = false
        settingsButton.setImage(UIImage(systemName: "gearshape.fill"), for: .normal)
        settingsButton.tintColor = .white
        settingsButton.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        settingsButton.layer.cornerRadius = 20
        settingsButton.addTarget(self, action: #selector(showSettings), for: .touchUpInside)
        view.addSubview(settingsButton)

        clearButton = UIButton(type: .system)
        clearButton.translatesAutoresizingMaskIntoConstraints = false
        clearButton.setImage(UIImage(systemName: "trash.fill"), for: .normal)
        clearButton.tintColor = .white
        clearButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.7)
        clearButton.layer.cornerRadius = 20
        clearButton.addTarget(self, action: #selector(clearAllBoxes), for: .touchUpInside)
        view.addSubview(clearButton)

        saveButton = UIButton(type: .system)
        saveButton.translatesAutoresizingMaskIntoConstraints = false
        saveButton.setImage(UIImage(systemName: "square.and.arrow.down.fill"), for: .normal)
        saveButton.tintColor = .white
        saveButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
        saveButton.layer.cornerRadius = 20
        saveButton.addTarget(self, action: #selector(saveCurrentFrame), for: .touchUpInside)
        view.addSubview(saveButton)

        deleteButton = UIButton(type: .system)
        deleteButton.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
        deleteButton.tintColor = .white
        deleteButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.85)
        deleteButton.layer.cornerRadius = 14
        deleteButton.frame = CGRect(x: 0, y: 0, width: 28, height: 28)
        deleteButton.addTarget(self, action: #selector(deleteSelectedBox), for: .touchUpInside)
        deleteButton.isHidden = true
        view.addSubview(deleteButton)

        pickPhotoButton = UIButton(type: .system)
        pickPhotoButton.translatesAutoresizingMaskIntoConstraints = false
        pickPhotoButton.setTitle("  Upload Image  ", for: .normal)
        pickPhotoButton.setImage(UIImage(systemName: "photo.on.rectangle.angled"), for: .normal)
        pickPhotoButton.tintColor = UIColor.darkGray
        pickPhotoButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        pickPhotoButton.setTitleColor(UIColor.darkGray, for: .normal)
        pickPhotoButton.backgroundColor = .white
        pickPhotoButton.layer.cornerRadius = 16
        pickPhotoButton.layer.shadowColor = UIColor.black.cgColor
        pickPhotoButton.layer.shadowOpacity = 0.3
        pickPhotoButton.layer.shadowOffset = CGSize(width: 0, height: 2)
        pickPhotoButton.layer.shadowRadius = 4
        pickPhotoButton.addTarget(self, action: #selector(pickPhotoTapped), for: .touchUpInside)
        pickPhotoButton.isHidden = true
        view.addSubview(pickPhotoButton)

        changePhotoButton = UIButton(type: .system)
        changePhotoButton.translatesAutoresizingMaskIntoConstraints = false
        changePhotoButton.setImage(UIImage(systemName: "square.and.arrow.up"), for: .normal)
        changePhotoButton.tintColor = .white
        changePhotoButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.8)
        changePhotoButton.layer.cornerRadius = 20
        changePhotoButton.addTarget(self, action: #selector(pickPhotoTapped), for: .touchUpInside)
        changePhotoButton.isHidden = true
        view.addSubview(changePhotoButton)

        // Prompt bar with class chips
        promptBar = UIView()
        promptBar.translatesAutoresizingMaskIntoConstraints = false
        promptBar.backgroundColor = UIColor.black.withAlphaComponent(0.75)
        promptBar.layer.cornerRadius = 12
        promptBar.clipsToBounds = true
        view.addSubview(promptBar)

        chipsScrollView = UIScrollView()
        chipsScrollView.translatesAutoresizingMaskIntoConstraints = false
        chipsScrollView.showsHorizontalScrollIndicator = false
        promptBar.addSubview(chipsScrollView)

        chipsStack = UIStackView()
        chipsStack.translatesAutoresizingMaskIntoConstraints = false
        chipsStack.axis = .horizontal
        chipsStack.spacing = 6
        chipsStack.alignment = .center
        chipsScrollView.addSubview(chipsStack)

        // Caption bar for selected box info
        captionLabel = UILabel()
        captionLabel.translatesAutoresizingMaskIntoConstraints = false
        captionLabel.textColor = .white
        captionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.75)
        captionLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        captionLabel.textAlignment = .center
        captionLabel.layer.cornerRadius = 10
        captionLabel.clipsToBounds = true
        captionLabel.isHidden = true
        view.addSubview(captionLabel)

        modeSegment = UISegmentedControl(items: ["RGB", "RGBD", "RGBD+SAM2"])
        modeSegment.translatesAutoresizingMaskIntoConstraints = false
        modeSegment.selectedSegmentIndex = 2
        modeSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        modeSegment.selectedSegmentTintColor = UIColor.systemBlue
        modeSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        modeSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        modeSegment.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        view.addSubview(modeSegment)

        detectMethodSegment = UISegmentedControl(items: ["Classes", "2D Box"])
        detectMethodSegment.translatesAutoresizingMaskIntoConstraints = false
        detectMethodSegment.selectedSegmentIndex = 0
        detectMethodSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        detectMethodSegment.selectedSegmentTintColor = UIColor.systemPurple
        detectMethodSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        detectMethodSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        detectMethodSegment.addTarget(self, action: #selector(detectMethodChanged), for: .valueChanged)
        view.addSubview(detectMethodSegment)

        inputModeSegment = UISegmentedControl(items: ["Camera", "Upload"])
        inputModeSegment.translatesAutoresizingMaskIntoConstraints = false
        inputModeSegment.selectedSegmentIndex = 0
        inputModeSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        inputModeSegment.selectedSegmentTintColor = UIColor.systemOrange
        inputModeSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        inputModeSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        inputModeSegment.addTarget(self, action: #selector(inputModeChanged), for: .valueChanged)
        view.addSubview(inputModeSegment)

        // Multi-View and Video modes disabled for now; code kept for future use
        captureSegment = UISegmentedControl(items: ["Single", "Multi-View", "Video"])
        captureSegment.translatesAutoresizingMaskIntoConstraints = false
        captureSegment.selectedSegmentIndex = 0
        captureSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        captureSegment.selectedSegmentTintColor = UIColor.systemOrange
        captureSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        captureSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        captureSegment.addTarget(self, action: #selector(captureSegmentChanged), for: .valueChanged)
        captureSegment.isHidden = true
        view.addSubview(captureSegment)

        NSLayoutConstraint.activate([
            // Top row: prompt bar + mode segment + settings gear
            promptBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            promptBar.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            promptBar.trailingAnchor.constraint(equalTo: modeSegment.leadingAnchor, constant: -8),
            promptBar.heightAnchor.constraint(equalToConstant: 40),

            chipsScrollView.topAnchor.constraint(equalTo: promptBar.topAnchor),
            chipsScrollView.bottomAnchor.constraint(equalTo: promptBar.bottomAnchor),
            chipsScrollView.leadingAnchor.constraint(equalTo: promptBar.leadingAnchor, constant: 6),
            chipsScrollView.trailingAnchor.constraint(equalTo: promptBar.trailingAnchor, constant: -6),

            chipsStack.topAnchor.constraint(equalTo: chipsScrollView.topAnchor),
            chipsStack.bottomAnchor.constraint(equalTo: chipsScrollView.bottomAnchor),
            chipsStack.leadingAnchor.constraint(equalTo: chipsScrollView.leadingAnchor),
            chipsStack.trailingAnchor.constraint(equalTo: chipsScrollView.trailingAnchor),
            chipsStack.heightAnchor.constraint(equalTo: chipsScrollView.heightAnchor),

            modeSegment.centerYAnchor.constraint(equalTo: promptBar.centerYAnchor),
            modeSegment.trailingAnchor.constraint(equalTo: settingsButton.leadingAnchor, constant: -8),
            modeSegment.heightAnchor.constraint(equalToConstant: 32),

            settingsButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            settingsButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            settingsButton.widthAnchor.constraint(equalToConstant: 40),
            settingsButton.heightAnchor.constraint(equalToConstant: 40),

            clearButton.topAnchor.constraint(equalTo: settingsButton.bottomAnchor, constant: 8),
            clearButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            clearButton.widthAnchor.constraint(equalToConstant: 40),
            clearButton.heightAnchor.constraint(equalToConstant: 40),

            saveButton.topAnchor.constraint(equalTo: clearButton.bottomAnchor, constant: 8),
            saveButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            saveButton.widthAnchor.constraint(equalToConstant: 40),
            saveButton.heightAnchor.constraint(equalToConstant: 40),

            // Status below top bar
            statusLabel.topAnchor.constraint(equalTo: promptBar.bottomAnchor, constant: 8),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 28),

            activityIndicator.centerYAnchor.constraint(equalTo: statusLabel.centerYAnchor),
            activityIndicator.trailingAnchor.constraint(equalTo: statusLabel.leadingAnchor, constant: -8),

            // Caption for selected box — aligned with bottom row
            captionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captionLabel.centerYAnchor.constraint(equalTo: detectMethodSegment.centerYAnchor),
            captionLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 28),

            // Bottom row: input mode + detect method + capture segment (left) + buttons (right)
            inputModeSegment.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -18),
            inputModeSegment.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            inputModeSegment.heightAnchor.constraint(equalToConstant: 32),

            detectMethodSegment.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -18),
            detectMethodSegment.leadingAnchor.constraint(equalTo: inputModeSegment.trailingAnchor, constant: 8),
            detectMethodSegment.heightAnchor.constraint(equalToConstant: 32),

            captureSegment.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -18),
            captureSegment.leadingAnchor.constraint(equalTo: detectMethodSegment.trailingAnchor, constant: 12),
            captureSegment.heightAnchor.constraint(equalToConstant: 32),

            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            captureButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            captureButton.widthAnchor.constraint(equalToConstant: 120),
            captureButton.heightAnchor.constraint(equalToConstant: 50),

            pickPhotoButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            pickPhotoButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            pickPhotoButton.heightAnchor.constraint(equalToConstant: 56),

            changePhotoButton.topAnchor.constraint(equalTo: saveButton.bottomAnchor, constant: 8),
            changePhotoButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            changePhotoButton.widthAnchor.constraint(equalToConstant: 40),
            changePhotoButton.heightAnchor.constraint(equalToConstant: 40),

            multiViewDetectButton.bottomAnchor.constraint(equalTo: captureButton.bottomAnchor),
            multiViewDetectButton.trailingAnchor.constraint(equalTo: captureButton.leadingAnchor, constant: -12),
            multiViewDetectButton.widthAnchor.constraint(equalToConstant: 120),
            multiViewDetectButton.heightAnchor.constraint(equalToConstant: 50),
        ])
    }

    private func updateStatus(_ text: String, showSpinner: Bool = false) {
        DispatchQueue.main.async {
            self.statusLabel.text = "  \(text)  "
            if showSpinner {
                self.activityIndicator.startAnimating()
            } else {
                self.activityIndicator.stopAnimating()
            }
        }
    }

    // MARK: - Input Mode Toggle

    @objc private func inputModeChanged() {
        inputMode = inputModeSegment.selectedSegmentIndex
        clearDrawnBoxes()
        deselectAllBoxes()
        bboxRenderer.clearBoxes()

        if inputMode == 0 {
            // Camera mode
            uploadImageView.isHidden = true
            uploadedImage = nil
            uploadImageView.image = nil
            pickPhotoButton.isHidden = true
            changePhotoButton.isHidden = true
            modeSegment.isHidden = false
            startARSession()

            if detectionMethod == 0 {
                resetCaptureButton()
                captureButton.isEnabled = true
                updateStatus("Ready - Tap Detect")
            } else {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = .systemBlue
                captureButton.isEnabled = false
                updateStatus("Draw a 2D box around the object, then tap Detect")
            }
        } else {
            // Upload mode
            sceneView.session.pause()
            uploadImageView.isHidden = false
            pickPhotoButton.isHidden = false
            modeSegment.isHidden = true

            captureButton.setTitle("  Detect  ", for: .normal)
            captureButton.backgroundColor = .systemBlue
            captureButton.isEnabled = false

            if detectionMethod == 1 {
                updateStatus("Select a photo, then draw 2D boxes")
            } else {
                updateStatus("Select a photo to detect")
            }
        }
        print("[InputMode] \(inputMode == 0 ? "Camera" : "Upload")")
    }

    // MARK: - Photo Picker

    @objc private func pickPhotoTapped() {
        pickPhoto()
    }

    private func pickPhoto() {
        var config = PHPickerConfiguration()
        config.selectionLimit = 1
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = self
        present(picker, animated: true)
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let result = results.first else { return }

        result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
            guard let self = self, let image = object as? UIImage else { return }
            let normalized = self.normalizeOrientation(image)
            // Pre-encode PNG on background thread so Detect is instant
            let pngData = self.resizeAndEncode(image: normalized)
            DispatchQueue.main.async {
                self.uploadedImage = normalized
                self.uploadPngData = pngData
                self.uploadImageView.image = normalized
                self.pickPhotoButton.isHidden = true
                self.changePhotoButton.isHidden = false
                self.clearDrawnBoxes()

                if self.detectionMethod == 1 {
                    self.captureButton.isEnabled = false
                    self.updateStatus("Photo loaded - draw 2D boxes, then tap Detect")
                } else {
                    self.captureButton.isEnabled = true
                    self.updateStatus("Photo loaded - tap Detect")
                }
            }
        }
    }

    private func normalizeOrientation(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }

    // MARK: - Save to Photos

    @objc private func saveCurrentFrame() {
        if inputMode == 0 {
            let image = sceneView.snapshot()
            UIImageWriteToSavedPhotosAlbum(image, self, #selector(saveCompleted(_:didFinishSavingWithError:contextInfo:)), nil)
        } else {
            guard uploadedImage != nil else {
                updateStatus("No image to save")
                return
            }
            // Render uploadImageView + wireframe overlay into one image,
            // then crop to the actual image display rect (removes black bars)
            let renderer = UIGraphicsImageRenderer(bounds: uploadImageView.bounds)
            let fullScreenshot = renderer.image { _ in
                uploadImageView.drawHierarchy(in: uploadImageView.bounds, afterScreenUpdates: true)
            }

            // Crop to image display rect (remove black bars)
            let displayRect = imageDisplayRect()
            let screenScale = UIScreen.main.scale
            let cropRect = CGRect(
                x: displayRect.origin.x * screenScale,
                y: displayRect.origin.y * screenScale,
                width: displayRect.width * screenScale,
                height: displayRect.height * screenScale
            )
            if let cgCropped = fullScreenshot.cgImage?.cropping(to: cropRect) {
                let cropped = UIImage(cgImage: cgCropped)
                UIImageWriteToSavedPhotosAlbum(cropped, self, #selector(saveCompleted(_:didFinishSavingWithError:contextInfo:)), nil)
            } else {
                UIImageWriteToSavedPhotosAlbum(fullScreenshot, self, #selector(saveCompleted(_:didFinishSavingWithError:contextInfo:)), nil)
            }
        }
    }

    @objc private func saveCompleted(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            updateStatus("Save failed: \(error.localizedDescription)")
        } else {
            updateStatus("Saved to Photos")
        }
    }

    // MARK: - Capture & Detect

    @objc private func captureAndDetect() {
        // Upload mode
        if inputMode == 1 {
            guard let image = uploadedImage else {
                pickPhoto()
                return
            }
            if detectionMethod == 1 {
                sendAllBox2D()
                return
            }
            guard !isDetecting else { return }
            // Use cached PNG (pre-encoded when photo was picked)
            let pngData = uploadPngData ?? resizeAndEncode(image: image)
            let (imgW, imgH) = resizedImageDimensions(image: image)
            isDetecting = true
            captureButton.isEnabled = false
            updateStatus("Detecting objects...", showSpinner: true)

            Task {
                do {
                    let result = try await DetectionService.shared.detectUpload(pngData: pngData, imageWidth: imgW, imageHeight: imgH)
                    await MainActor.run { handleResult(result) }
                } catch {
                    await MainActor.run {
                        updateStatus("Error: \(error.localizedDescription)")
                        finishDetection()
                    }
                }
            }
            return
        }

        // Camera mode
        if detectionMethod == 1 {
            sendAllBox2D()
            return
        }

        // Multi-view allows tapping again while "detecting" (collecting frames)
        if captureMode == 1 && !multiViewFrames.isEmpty {
            captureMultiViewFrame()
            return
        }

        guard !isDetecting else { return }
        guard sceneView.session.currentFrame != nil else {
            updateStatus("AR session not ready")
            return
        }

        isDetecting = true
        captureButton.isEnabled = false

        switch captureMode {
        case 1:
            startMultiViewCapture()
        case 2:
            startVideoCapture()
        default:
            singleCapture()
        }
    }

    private func singleCapture() {
        guard let frame = sceneView.session.currentFrame else { return }
        updateStatus("Capturing...", showSpinner: true)
        flashScreen()

        let captureData = extractCaptureData(from: frame)
        updateStatus("Detecting objects...", showSpinner: true)

        Task {
            do {
                let result: DetectionResponse
                if useLocalModel {
                    if !LocalInferenceService.shared.isModelLoaded {
                        // Download model if needed, then load
                        try await LocalInferenceService.shared.downloadModelIfNeeded { status in
                            Task { @MainActor in self.updateStatus(status, showSpinner: true) }
                        }
                        await MainActor.run { updateStatus("Loading ONNX model...", showSpinner: true) }
                        try LocalInferenceService.shared.loadModels()
                    }
                    result = try await LocalInferenceService.shared.detect(capture: captureData)
                } else {
                    result = try await DetectionService.shared.detect(capture: captureData)
                }
                await MainActor.run { handleResult(result) }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    // MARK: - Multi-View Capture

    private func startMultiViewCapture() {
        multiViewFrames = []
        guard let frame = sceneView.session.currentFrame else { return }
        flashScreen()
        let data = extractCaptureData(from: frame)
        multiViewFrames.append(data)
        print("[MultiView] Captured main frame")

        updateStatus("Main captured — move to a different angle and tap + Ref")
        captureButton.setTitle("  + Ref  ", for: .normal)
        captureButton.backgroundColor = UIColor.systemTeal
        captureButton.isEnabled = true
    }

    private func captureMultiViewFrame() {
        guard let frame = sceneView.session.currentFrame else { return }
        flashScreen()
        let data = extractCaptureData(from: frame)
        multiViewFrames.append(data)
        let refCount = multiViewFrames.count - 1
        print("[MultiView] Captured ref \(refCount)")

        multiViewDetectButton.isHidden = false
        updateStatus("Main + \(refCount) ref(s) — tap + Ref for more, or Detect to send")
    }

    @objc private func sendMultiViewDetection() {
        guard multiViewFrames.count >= 2 else { return }
        captureButton.isEnabled = false
        multiViewDetectButton.isHidden = true

        let frames = multiViewFrames
        multiViewFrames = []

        let mainFrame = frames[0]
        let refFrames = Array(frames[1...])

        updateStatus("Detecting (main + \(refFrames.count) ref)...", showSpinner: true)
        print("[MultiView] Sending main + \(refFrames.count) ref frame(s) to API")

        Task {
            do {
                let result = try await DetectionService.shared.detectWithReferences(main: mainFrame, references: refFrames)
                await MainActor.run {
                    handleResult(result)
                    resetCaptureButton()
                }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                    resetCaptureButton()
                }
            }
        }
    }

    private func resetCaptureButton() {
        let titles = ["  Detect  ", "  Main  ", "  Record  "]
        let colors = [UIColor.systemBlue, UIColor.systemGreen, UIColor.systemRed]
        captureButton.setTitle(titles[captureMode], for: .normal)
        captureButton.backgroundColor = colors[captureMode]
        multiViewDetectButton.isHidden = true
    }

    // MARK: - Video Capture

    private func startVideoCapture() {
        videoFrames = []
        updateStatus("Recording 0/\(videoFrameCount)...", showSpinner: true)
        flashScreen()

        captureVideoFrame()

        videoTimer = Timer.scheduledTimer(withTimeInterval: videoFrameInterval, repeats: true) { [weak self] timer in
            guard let self = self else { timer.invalidate(); return }
            self.captureVideoFrame()

            if self.videoFrames.count >= self.videoFrameCount {
                timer.invalidate()
                self.videoTimer = nil
                self.finishVideoCapture()
            }
        }
    }

    private func captureVideoFrame() {
        guard let frame = sceneView.session.currentFrame else { return }
        let data = extractCaptureData(from: frame)
        videoFrames.append(data)
        let count = videoFrames.count
        print("[Video] Captured frame \(count)/\(videoFrameCount)")
        updateStatus("Recording \(count)/\(videoFrameCount)...", showSpinner: true)
    }

    private func finishVideoCapture() {
        let frames = videoFrames
        videoFrames = []

        updateStatus("Detecting (\(frames.count) frames)...", showSpinner: true)
        print("[Video] Sending \(frames.count) frames to API")

        Task {
            do {
                let result = try await DetectionService.shared.detectMultiFrame(captures: frames)
                await MainActor.run { handleResult(result) }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    // MARK: - Result Handling

    private func handleResult(_ result: DetectionResponse) {
        let modeTag = "[\(result.mode ?? "?")]"
        if result.boxes.isEmpty {
            updateStatus("\(modeTag) No objects detected")
        } else {
            let summary = result.boxes.map { box -> String in
                let pct = String(format: "%.0f%%", box.score * 100)
                if let nf = box.n_frames {
                    return "\(box.label)(\(pct),\(nf)f)"
                }
                return "\(box.label)(\(pct))"
            }.joined(separator: ", ")
            updateStatus("\(modeTag) Found \(result.boxes.count): \(summary)")
            if inputMode == 1 {
                // Upload mode: draw 2D wireframe overlay on image
                drawProjectedBoxes(result)
            } else {
                bboxRenderer.addBoxes(result.boxes)
            }
        }
        finishDetection()
    }

    private func finishDetection() {
        isDetecting = false

        if inputMode == 1 {
            // Upload mode
            if detectionMethod == 1 {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = .systemBlue
                captureButton.isEnabled = !drawnBoxCoords.isEmpty
                if drawnBoxCoords.isEmpty && uploadedImage != nil {
                    updateStatus("Draw a 2D box around the object, then tap Detect")
                }
            } else {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = .systemBlue
                captureButton.isEnabled = uploadedImage != nil
            }
            return
        }

        // Camera mode
        if detectionMethod == 1 {
            captureButton.setTitle("  Detect  ", for: .normal)
            captureButton.backgroundColor = UIColor.systemBlue
            captureButton.isEnabled = !drawnBoxCoords.isEmpty
            if drawnBoxCoords.isEmpty {
                updateStatus("Draw a 2D box around the object, then tap Detect")
            }
        } else {
            captureButton.isEnabled = true
            resetCaptureButton()
        }
    }

    private func flashScreen() {
        let flash = UIView(frame: view.bounds)
        flash.backgroundColor = .white
        flash.alpha = 0.8
        view.addSubview(flash)
        UIView.animate(withDuration: 0.2) {
            flash.alpha = 0
        } completion: { _ in
            flash.removeFromSuperview()
        }
    }

    // MARK: - Real-time Label Updates

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard inputMode == 0 else { return }
        guard let pov = sceneView.pointOfView else { return }
        bboxRenderer.updateLabels(cameraPosition: pov.simdWorldPosition, unitMode: unitMode)

        if let selected = bboxRenderer.selectedNode {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.updateDeleteButtonForSelected3D()
                if let text = self.bboxRenderer.captionText(for: selected, cameraPosition: pov.simdWorldPosition, unitMode: self.unitMode) {
                    self.captionLabel.text = "  \(text)  "
                    self.captionLabel.isHidden = false
                }
            }
        }
    }

    // MARK: - Clear Boxes

    @objc private func clearAllBoxes() {
        let count2D = drawnBoxViews.count
        let count3D = bboxRenderer.boxCount
        clearDrawnBoxes()
        bboxRenderer.clearBoxes()
        updateStatus("\(count3D) 3D + \(count2D) 2D box(es) cleared")
    }

    // MARK: - 2D Box Drawing (Multiple)

    @objc private func handleBoxDrag(_ gesture: UIPanGestureRecognizer) {
        // Upload mode: use uploadImageView coords so displayRect mapping is correct
        let targetView: UIView = inputMode == 1 ? (uploadImageView ?? view) : view
        let location = gesture.location(in: targetView)

        switch gesture.state {
        case .began:
            deselectAllBoxes()
            dragStartPoint = location

            let overlay = makeBoxOverlay()
            overlay.frame = CGRect(origin: location, size: .zero)
            overlay.isHidden = false
            targetView.addSubview(overlay)
            targetView.bringSubviewToFront(overlay)
            drawnBoxViews.append(overlay)

        case .changed:
            guard let start = dragStartPoint, let overlay = drawnBoxViews.last else { return }
            let rect = CGRect(
                x: min(start.x, location.x),
                y: min(start.y, location.y),
                width: abs(location.x - start.x),
                height: abs(location.y - start.y)
            )
            overlay.frame = rect

        case .ended, .cancelled:
            guard let start = dragStartPoint else { return }
            dragStartPoint = nil
            let endPoint = location

            let minDrag: CGFloat = 8
            if abs(endPoint.x - start.x) < minDrag || abs(endPoint.y - start.y) < minDrag {
                drawnBoxViews.last?.removeFromSuperview()
                drawnBoxViews.removeLast()
                return
            }

            storeBox2DCoords(screenStart: start, screenEnd: endPoint)

        default:
            break
        }
    }

    private func makeBoxOverlay() -> UIView {
        let v = UIView()
        v.backgroundColor = UIColor.systemYellow.withAlphaComponent(0.15)
        v.layer.borderColor = UIColor.systemYellow.cgColor
        v.layer.borderWidth = 2.5
        v.isUserInteractionEnabled = false
        return v
    }

    private func storeBox2DCoords(screenStart: CGPoint, screenEnd: CGPoint) {
        if inputMode == 1 {
            storeBox2DCoordsUpload(screenStart: screenStart, screenEnd: screenEnd)
            return
        }

        guard let frame = sceneView.session.currentFrame else { return }

        if box2DCaptureData == nil {
            flashScreen()
            box2DCaptureData = extractCaptureData(from: frame)
        }

        let viewSize = sceneView.bounds.size
        let dt = frame.displayTransform(for: .landscapeRight, viewportSize: viewSize)
        let inv = dt.inverted()

        let normStart = CGPoint(x: screenStart.x / viewSize.width, y: screenStart.y / viewSize.height)
        let normEnd = CGPoint(x: screenEnd.x / viewSize.width, y: screenEnd.y / viewSize.height)
        let imgStart = normStart.applying(inv)
        let imgEnd = normEnd.applying(inv)

        let pixelBuffer = frame.capturedImage
        let origW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let origH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let imgW = origW * scale
        let imgH = origH * scale

        let x1 = Float(min(imgStart.x, imgEnd.x) * imgW)
        let y1 = Float(min(imgStart.y, imgEnd.y) * imgH)
        let x2 = Float(max(imgStart.x, imgEnd.x) * imgW)
        let y2 = Float(max(imgStart.y, imgEnd.y) * imgH)
        let box2D = [max(0, x1), max(0, y1), min(Float(imgW), x2), min(Float(imgH), y2)]

        drawnBoxCoords.append(box2D)
        captureButton.isEnabled = true

        print("[Box2D] Stored box #\(drawnBoxCoords.count): [\(box2D[0]), \(box2D[1]), \(box2D[2]), \(box2D[3])] in \(Int(imgW))x\(Int(imgH))")
        updateStatus("\(drawnBoxCoords.count) box(es) drawn — draw more or tap Detect")
    }

    private func storeBox2DCoordsUpload(screenStart: CGPoint, screenEnd: CGPoint) {
        guard let image = uploadedImage else { return }

        if box2DCaptureData == nil && inputMode == 0 {
            // Camera mode: capture AR frame data
            flashScreen()
        }
        // Upload mode: no CaptureData needed, will use detectUploadWithBox2D
        if box2DCaptureData == nil && inputMode == 1 {
            flashScreen()
            // Store PNG data for upload (no intrinsics/pose needed)
            uploadPngData = resizeAndEncode(image: image)
        }

        let displayRect = imageDisplayRect()

        let normStartX = (screenStart.x - displayRect.origin.x) / displayRect.width
        let normStartY = (screenStart.y - displayRect.origin.y) / displayRect.height
        let normEndX = (screenEnd.x - displayRect.origin.x) / displayRect.width
        let normEndY = (screenEnd.y - displayRect.origin.y) / displayRect.height

        let origW = image.size.width
        let origH = image.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let imgW = origW * scale
        let imgH = origH * scale

        let x1 = Float(min(normStartX, normEndX) * imgW)
        let y1 = Float(min(normStartY, normEndY) * imgH)
        let x2 = Float(max(normStartX, normEndX) * imgW)
        let y2 = Float(max(normStartY, normEndY) * imgH)
        let box2D = [max(0, x1), max(0, y1), min(Float(imgW), x2), min(Float(imgH), y2)]

        drawnBoxCoords.append(box2D)
        captureButton.isEnabled = true

        print("[Box2D-Upload] Stored box #\(drawnBoxCoords.count): \(box2D) in \(Int(imgW))x\(Int(imgH))")
        updateStatus("\(drawnBoxCoords.count) box(es) drawn — draw more or tap Detect")
    }

    // MARK: - Upload Mode 2D Wireframe Rendering

    private var wireframeOverlay: UIView?

    private func drawProjectedBoxes(_ result: DetectionResponse) {
        // Remove old overlay
        wireframeOverlay?.removeFromSuperview()

        guard let image = uploadedImage else { return }
        let displayRect = imageDisplayRect()
        let boxes = result.boxes

        // Determine the API's image dimensions from predicted_intrinsics
        // The principal point (cx, cy) ≈ center of image, so apiW ≈ 2*cx, apiH ≈ 2*cy
        let origW = image.size.width
        let origH = image.size.height
        let resizeScale = min(1.0, maxImageDimension / max(origW, origH))
        let sentW = Float(origW * resizeScale)
        let sentH = Float(origH * resizeScale)

        // Use OUR sent intrinsics (matching the identity cam2world we sent to API)
        let fx = max(sentW, sentH)
        let fy = fx
        let pcx = sentW / 2
        let pcy = sentH / 2
        print("[Upload] Using sent intrinsics: f=\(fx), cx=\(pcx), cy=\(pcy), sent=\(sentW)x\(sentH)")

        let overlay = UIView(frame: uploadImageView.bounds)
        overlay.backgroundColor = .clear
        overlay.isUserInteractionEnabled = false

        let edges: [(Int, Int)] = [
            (0,1),(2,3),(4,5),(6,7),
            (0,2),(1,3),(4,6),(5,7),
            (0,4),(1,5),(2,6),(3,7),
        ]

        var drawnCount = 0
        for box in boxes {
            // Client-side reprojection with coordinate system conversion
            // API uses left-handed (Unity) convention; convert like BBoxRenderer does:
            //   center.z negated, quaternion (ix,iy negated, iz kept), extra 90° Y rotation
            let rawCenter = box.centerSIMD
            let center = simd_float3(rawCenter.x, rawCenter.y, -rawCenter.z)

            let q = box.quaternion
            let rhQuat = simd_quatf(ix: -q.imag.x, iy: -q.imag.y, iz: q.imag.z, r: q.real)
            let finalQuat = rhQuat * simd_quatf(angle: .pi / 2, axis: simd_float3(0, 1, 0))

            let size = box.sizeSIMD
            let hx = size.x / 2, hy = size.y / 2, hz = size.z / 2
            let localCorners: [simd_float3] = [
                simd_float3(-hx, -hy, -hz), simd_float3( hx, -hy, -hz),
                simd_float3(-hx,  hy, -hz), simd_float3( hx,  hy, -hz),
                simd_float3(-hx, -hy,  hz), simd_float3( hx, -hy,  hz),
                simd_float3(-hx,  hy,  hz), simd_float3( hx,  hy,  hz),
            ]

            var screenPts: [CGPoint] = []
            var allValid = true
            for lc in localCorners {
                // Rotate local corner, then translate to world
                let world = finalQuat.act(lc) + center
                // For pinhole projection, need positive depth
                // In right-handed camera space: -Z is forward, so depth = -world.z
                let depth = -world.z
                guard depth > 0.001 else { allValid = false; break }
                let u = fx * world.x / depth + pcx
                // Y-up in 3D → Y-down in image: v = cy - fy * Y / depth
                let v = pcy - fy * world.y / depth
                let normU = CGFloat(u) / CGFloat(sentW)
                let normV = CGFloat(v) / CGFloat(sentH)
                let sx = displayRect.origin.x + normU * displayRect.width
                let sy = displayRect.origin.y + normV * displayRect.height
                screenPts.append(CGPoint(x: sx, y: sy))
            }
            if !allValid { screenPts.removeAll() }

            guard screenPts.count == 8 else {
                print("[Upload] Box \(box.label): skipped (invalid depth)")
                continue
            }

            let c = box.displayColor
            let color = UIColor(red: CGFloat(c.r), green: CGFloat(c.g), blue: CGFloat(c.b), alpha: 1.0)

            // Draw 12 edges
            for (i, j) in edges {
                let line = CAShapeLayer()
                let path = UIBezierPath()
                path.move(to: screenPts[i])
                path.addLine(to: screenPts[j])
                line.path = path.cgPath
                line.strokeColor = color.cgColor
                line.lineWidth = 2.0
                line.fillColor = nil
                overlay.layer.addSublayer(line)
            }

            // Label at topmost corner
            if let topPt = screenPts.min(by: { $0.y < $1.y }) {
                let label = UILabel(frame: CGRect(x: topPt.x - 50, y: topPt.y - 20, width: 120, height: 16))
                label.text = "\(box.label) \(String(format: "%.0f%%", box.score * 100))"
                label.font = .systemFont(ofSize: 11, weight: .bold)
                label.textColor = color
                label.textAlignment = .center
                label.backgroundColor = UIColor.black.withAlphaComponent(0.5)
                label.layer.cornerRadius = 3
                label.clipsToBounds = true
                overlay.addSubview(label)
            }
            drawnCount += 1
        }

        uploadImageView.addSubview(overlay)
        wireframeOverlay = overlay
        print("[Upload] Drew \(drawnCount)/\(boxes.count) 2D wireframe boxes, displayRect=\(displayRect), viewBounds=\(uploadImageView.bounds)")
    }

    private func imageDisplayRect() -> CGRect {
        guard let image = uploadedImage else { return uploadImageView.bounds }
        let viewSize = uploadImageView.bounds.size
        let imageSize = image.size
        let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let scaledW = imageSize.width * scale
        let scaledH = imageSize.height * scale
        return CGRect(
            x: (viewSize.width - scaledW) / 2,
            y: (viewSize.height - scaledH) / 2,
            width: scaledW,
            height: scaledH
        )
    }

    private func sendAllBox2D() {
        guard !drawnBoxCoords.isEmpty else { return }
        // Camera mode needs CaptureData, upload mode needs pngData
        if inputMode == 0 && box2DCaptureData == nil { return }
        if inputMode == 1 && uploadPngData == nil { return }
        guard !isDetecting else { return }

        isDetecting = true
        captureButton.isEnabled = false
        let count = drawnBoxCoords.count
        updateStatus("Detecting \(count) box(es)...", showSpinner: true)

        let coordsCopy = drawnBoxCoords
        let captureCopy = box2DCaptureData
        let pngCopy = uploadPngData
        let isUpload = inputMode == 1
        let uploadImgSize = uploadedImage.map { resizedImageDimensions(image: $0) }

        Task {
            var allBoxes: [BBox3D] = []
            var lastMode: String?
            var errorMsg: String?

            await withTaskGroup(of: (DetectionResponse?, Error?).self) { group in
                for (i, box2D) in coordsCopy.enumerated() {
                    group.addTask {
                        do {
                            print("[Box2D] Sending box \(i+1)/\(count)")
                            let result: DetectionResponse
                            if isUpload, let png = pngCopy, let dims = uploadImgSize {
                                result = try await DetectionService.shared.detectUploadWithBox2D(pngData: png, box2D: box2D, imageWidth: dims.0, imageHeight: dims.1)
                            } else if let capture = captureCopy {
                                result = try await DetectionService.shared.detectWithBox2D(capture: capture, box2D: box2D)
                            } else {
                                throw DetectionError.serverError(statusCode: -1, message: "No capture data")
                            }
                            return (result, nil)
                        } catch {
                            return (nil, error)
                        }
                    }
                }
                for await (result, error) in group {
                    if let r = result {
                        allBoxes.append(contentsOf: r.boxes)
                        lastMode = r.mode
                    } else if let e = error, errorMsg == nil {
                        errorMsg = e.localizedDescription
                    }
                }
            }

            await MainActor.run {
                clearDrawnBoxes()
                if !allBoxes.isEmpty {
                    let combined = DetectionResponse(boxes: allBoxes, mode: lastMode, predicted_intrinsics: nil)
                    handleResult(combined)
                } else if let err = errorMsg {
                    updateStatus("Error: \(err)")
                    finishDetection()
                } else {
                    updateStatus("No objects detected from \(count) box(es)")
                    finishDetection()
                }
            }
        }
    }

    private func clearDrawnBoxes() {
        for v in drawnBoxViews { v.removeFromSuperview() }
        drawnBoxViews.removeAll()
        drawnBoxCoords.removeAll()
        box2DCaptureData = nil
        uploadPngData = nil
        selectedBox2DIndex = nil
        deleteButton.isHidden = true
        wireframeOverlay?.removeFromSuperview()
        wireframeOverlay = nil
    }

    // MARK: - Box Selection & Deletion

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        let point = gesture.location(in: view)

        // Check 2D boxes first (only in 2D Box mode)
        if detectionMethod == 1 {
            for (i, boxView) in drawnBoxViews.enumerated().reversed() {
                if boxView.frame.contains(point) {
                    selectBox2D(at: i)
                    return
                }
            }
        }

        // Check 3D boxes (camera mode only)
        if inputMode == 0 {
            let scenePoint = gesture.location(in: sceneView)
            if let boxNode = bboxRenderer.nearestBox(to: scenePoint, in: sceneView) {
                selectBox3D(boxNode)
                return
            }
        }

        // Tap on empty → deselect
        deselectAllBoxes()
    }

    private func selectBox2D(at index: Int) {
        deselectAllBoxes()
        selectedBox2DIndex = index
        let boxView = drawnBoxViews[index]
        boxView.layer.borderColor = UIColor.systemRed.cgColor
        boxView.backgroundColor = UIColor.systemRed.withAlphaComponent(0.25)

        captionLabel.text = "  2D Box #\(index + 1) — Tap Detect to identify  "
        captionLabel.isHidden = false

        deleteButton.isHidden = false
        positionDeleteButton(near: boxView.frame)
    }

    private func selectBox3D(_ node: SCNNode) {
        deselectAllBoxes()
        bboxRenderer.selectBox(node)
        updateDeleteButtonForSelected3D()
        if let pov = sceneView.pointOfView,
           let text = bboxRenderer.captionText(for: node, cameraPosition: pov.simdWorldPosition, unitMode: unitMode) {
            captionLabel.text = "  \(text)  "
            captionLabel.isHidden = false
        }
    }

    private func deselectAllBoxes() {
        if let idx = selectedBox2DIndex, idx < drawnBoxViews.count {
            let boxView = drawnBoxViews[idx]
            boxView.layer.borderColor = UIColor.systemYellow.cgColor
            boxView.backgroundColor = UIColor.systemYellow.withAlphaComponent(0.15)
        }
        selectedBox2DIndex = nil
        bboxRenderer.deselectAll()
        deleteButton.isHidden = true
        captionLabel.isHidden = true
    }

    private func positionDeleteButton(near rect: CGRect) {
        deleteButton.center = CGPoint(x: rect.maxX - 2, y: rect.minY + 2)
        view.bringSubviewToFront(deleteButton)
    }

    private func updateDeleteButtonForSelected3D() {
        guard let node = bboxRenderer.selectedNode else {
            deleteButton.isHidden = true
            return
        }
        let screenPos = sceneView.projectPoint(SCNVector3(node.simdWorldPosition))
        if screenPos.z > 0 && screenPos.z < 1 {
            deleteButton.isHidden = false
            deleteButton.center = CGPoint(
                x: CGFloat(screenPos.x) + 20,
                y: CGFloat(screenPos.y) - 20
            )
            view.bringSubviewToFront(deleteButton)
        } else {
            deleteButton.isHidden = true
        }
    }

    @objc private func deleteSelectedBox() {
        if let idx = selectedBox2DIndex, idx < drawnBoxViews.count {
            drawnBoxViews[idx].removeFromSuperview()
            drawnBoxViews.remove(at: idx)
            drawnBoxCoords.remove(at: idx)
            selectedBox2DIndex = nil
            deleteButton.isHidden = true
            if drawnBoxViews.isEmpty {
                box2DCaptureData = nil
                captureButton.isEnabled = false
                updateStatus("Draw a 2D box around the object, then tap Detect")
            } else {
                updateStatus("\(drawnBoxCoords.count) box(es) drawn — draw more or tap Detect")
            }
            return
        }

        if let node = bboxRenderer.selectedNode {
            bboxRenderer.removeBox(node)
            deleteButton.isHidden = true
            updateStatus("\(bboxRenderer.boxCount) 3D box(es) remaining")
        }
    }

    // MARK: - Extract ARFrame Data

    /// Max dimension for the image sent to the API. iPhone cameras are 4032x3024 which is too large.
    private let maxImageDimension: CGFloat = 1920

    private func extractCaptureData(from frame: ARFrame) -> CaptureData {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
        let fullImage = UIImage(cgImage: cgImage)

        let origW = fullImage.size.width
        let origH = fullImage.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let newW = origW * scale
        let newH = origH * scale

        let resized: UIImage
        if scale < 1.0 {
            let renderer = UIGraphicsImageRenderer(size: CGSize(width: newW, height: newH))
            resized = renderer.image { _ in
                fullImage.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
            }
        } else {
            resized = fullImage
        }
        let pngData = resized.pngData()!

        let intrinsics = frame.camera.intrinsics
        let fx = intrinsics.columns.0.x * Float(scale)
        let fy = intrinsics.columns.1.y * Float(scale)
        let cx = intrinsics.columns.2.x * Float(scale)
        let cyRaw = intrinsics.columns.2.y * Float(scale)
        // ARKit cy is from top edge (OpenCV). API expects from bottom edge (Unity).
        let cy = Float(newH) - cyRaw
        let K: [[Float]] = [
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ]

        // Convert ARKit right-handed cam2world to Unity left-handed: M_lh = S * M_rh * S
        // where S = diag(1,1,-1). Negate Z column + Z row of rotation + tz. det stays +1.
        let t = frame.camera.transform
        let camToWorld: [[Float]] = [
            [ t.columns.0.x,  t.columns.1.x, -t.columns.2.x,  t.columns.3.x],
            [ t.columns.0.y,  t.columns.1.y, -t.columns.2.y,  t.columns.3.y],
            [-t.columns.0.z, -t.columns.1.z,  t.columns.2.z, -t.columns.3.z],
            [0, 0, 0, 1],
        ]

        var depthPng: Data? = nil
        if sendDepth, let sceneDepth = frame.sceneDepth {
            depthPng = depthMapTo16BitPNG(sceneDepth.depthMap)
            print("[Capture] Depth: \(CVPixelBufferGetWidth(sceneDepth.depthMap))x\(CVPixelBufferGetHeight(sceneDepth.depthMap)), PNG: \(depthPng?.count ?? 0)B")
        } else {
            print("[Capture] Depth skipped (mode=\(sendDepth ? "depth" : "rgb-only"), hasLiDAR=\(frame.sceneDepth != nil))")
        }

        print("[Capture] Original: \(Int(origW))x\(Int(origH)), Resized: \(Int(newW))x\(Int(newH)), PNG: \(pngData.count/1024)KB")
        print("[Capture] K: fx=\(fx) fy=\(fy) cx=\(cx) cy=\(cy) (raw=\(cyRaw), H=\(Int(newH)))")

        return CaptureData(pngData: pngData, depthPngData: depthPng, alignmentPostProcess: alignmentPostProcess, intrinsicK: K, cameraToWorld: camToWorld)
    }

    /// Resize image and encode to PNG for upload mode (no intrinsics/pose needed)
    private func resizeAndEncode(image: UIImage) -> Data {
        let origW = image.size.width
        let origH = image.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let newW = origW * scale
        let newH = origH * scale

        // Force standard sRGB 8-bit at @1x scale so pixel dimensions = point dimensions
        let format = UIGraphicsImageRendererFormat()
        format.preferredRange = .standard
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: newW, height: newH), format: format)
        let resized = renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
        }
        // Use JPEG (quality 85%) instead of PNG — ~10x smaller for photos
        let jpegData = resized.jpegData(compressionQuality: 0.85)!
        print("[Upload] Resized: \(Int(newW))x\(Int(newH)), JPEG: \(jpegData.count/1024)KB")
        return jpegData
    }

    private func resizedImageDimensions(image: UIImage) -> (Int, Int) {
        let origW = image.size.width
        let origH = image.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        return (Int(origW * scale), Int(origH * scale))
    }

    @available(*, deprecated, message: "Use resizeAndEncode + detectUpload instead")
    private func extractCaptureDataFromUpload(image: UIImage) -> CaptureData {
        let pngData = resizeAndEncode(image: image)
        let origW = image.size.width
        let origH = image.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let newW = origW * scale
        let newH = origH * scale

        let f = Float(max(newW, newH))
        let cx = Float(newW) / 2
        let cy = Float(newH) / 2
        let K: [[Float]] = [
            [f,  0, cx],
            [0,  f, cy],
            [0,  0,  1],
        ]

        // Identity camera-to-world
        let camToWorld: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        print("[Upload] Original: \(Int(origW))x\(Int(origH)), Resized: \(Int(newW))x\(Int(newH)), PNG: \(pngData.count/1024)KB")
        print("[Upload] Default K: f=\(f) cx=\(cx) cy=\(cy)")

        return CaptureData(pngData: pngData, depthPngData: nil, alignmentPostProcess: false, intrinsicK: K, cameraToWorld: camToWorld)
    }

    private func depthMapTo16BitPNG(_ depthMap: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let floatPtr = base.assumingMemoryBound(to: Float32.self)
        let stride = rowBytes / 4

        var uint16Data = [UInt16](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let meters = floatPtr[y * stride + x]
                if meters > 0 && meters < 65.535 {
                    uint16Data[y * w + x] = UInt16(meters * 1000.0)
                }
            }
        }

        return uint16Data.withUnsafeMutableBytes { rawBuf -> Data? in
            guard let ptr = rawBuf.baseAddress else { return nil }
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Little.rawValue)
            guard let provider = CGDataProvider(data: NSData(bytes: ptr, length: w * h * 2)),
                  let cgImage = CGImage(width: w, height: h,
                                        bitsPerComponent: 16, bitsPerPixel: 16,
                                        bytesPerRow: w * 2,
                                        space: CGColorSpaceCreateDeviceGray(),
                                        bitmapInfo: bitmapInfo,
                                        provider: provider,
                                        decode: nil, shouldInterpolate: false,
                                        intent: .defaultIntent)
            else { return nil }

            let mutableData = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cgImage, nil)
            guard CGImageDestinationFinalize(dest) else { return nil }
            return mutableData as Data
        }
    }

    // MARK: - Detection Method Toggle

    @objc private func detectMethodChanged() {
        detectionMethod = detectMethodSegment.selectedSegmentIndex
        clearDrawnBoxes()
        deselectAllBoxes()

        let isBox = detectionMethod == 1
        panGesture.isEnabled = isBox
        promptBar.isHidden = isBox
        multiViewDetectButton.isHidden = true
        multiViewFrames = []

        if inputMode == 1 {
            // Upload mode
            if isBox {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = .systemBlue
                captureButton.isEnabled = false
                if uploadedImage != nil {
                    updateStatus("Draw a 2D box around the object, then tap Detect")
                } else {
                    updateStatus("Select a photo first, then draw 2D boxes")
                }
            } else {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = .systemBlue
                captureButton.isEnabled = uploadedImage != nil
                updateStatus(uploadedImage != nil ? "Ready - Tap Detect" : "Select a photo to detect")
            }
        } else {
            // Camera mode
            if isBox {
                captureButton.setTitle("  Detect  ", for: .normal)
                captureButton.backgroundColor = UIColor.systemBlue
                captureButton.isEnabled = false
                updateStatus("Draw a 2D box around the object, then tap Detect")
            } else {
                resetCaptureButton()
                captureButton.isEnabled = true
                updateStatus("Ready - Tap Detect")
            }
        }
        print("[DetectMethod] \(isBox ? "2D Box" : "Classes")")
    }

    // MARK: - Capture Mode Toggle

    @objc private func captureSegmentChanged() {
        captureMode = captureSegment.selectedSegmentIndex
        multiViewFrames = []
        resetCaptureButton()
        let modeNames = ["Single", "Multi-View", "Video (\(videoFrameCount) frames)"]
        print("[CaptureMode] \(modeNames[captureMode])")
    }

    // MARK: - Depth Mode Toggle

    @objc private func modeChanged() {
        let idx = modeSegment.selectedSegmentIndex
        sendDepth = idx >= 1
        alignmentPostProcess = idx == 2
        let modes = ["RGB", "RGBD", "RGBD+SAM2"]
        print("[Mode] Switched to \(modes[idx])")
        if sendDepth && !isLiDARAvailable {
            updateStatus("No LiDAR on this device - will use monocular")
        }
    }

    // MARK: - Class Management

    private func loadClasses() {
        let defaults = UserDefaults.standard
        savedClasses = defaults.stringArray(forKey: "savedClasses") ?? []
        let active = defaults.stringArray(forKey: "activeClasses") ?? []
        activeClasses = Set(active)

        if savedClasses.isEmpty {
            resetToDefaultClasses()
        }
        updateTextPrompt()
    }

    private func saveClasses() {
        let defaults = UserDefaults.standard
        defaults.set(savedClasses, forKey: "savedClasses")
        defaults.set(Array(activeClasses), forKey: "activeClasses")
        updateTextPrompt()
    }

    private func updateTextPrompt() {
        let active = savedClasses.filter { activeClasses.contains($0) }
        DetectionService.shared.textPrompt = active.joined(separator: ".")
        print("[Classes] Active: \(active.joined(separator: ", "))")
    }

    private static let defaultClasses = ["cup", "car", "table", "chair", "person"]

    private func resetToDefaultClasses() {
        savedClasses = Self.defaultClasses
        activeClasses = Set(Self.defaultClasses)
        saveClasses()
    }

    private func rebuildChips() {
        chipsStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        for className in savedClasses {
            let chip = makeChip(title: className, isActive: activeClasses.contains(className))
            chipsStack.addArrangedSubview(chip)
        }
        let addBtn = UIButton(type: .system)
        addBtn.setTitle(" + ", for: .normal)
        addBtn.titleLabel?.font = .systemFont(ofSize: 16, weight: .bold)
        addBtn.setTitleColor(.white, for: .normal)
        addBtn.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
        addBtn.layer.cornerRadius = 14
        addBtn.widthAnchor.constraint(equalToConstant: 36).isActive = true
        addBtn.addTarget(self, action: #selector(addNewClass), for: .touchUpInside)
        chipsStack.addArrangedSubview(addBtn)

        let resetBtn = UIButton(type: .system)
        resetBtn.setImage(UIImage(systemName: "arrow.counterclockwise"), for: .normal)
        resetBtn.tintColor = .lightGray
        resetBtn.widthAnchor.constraint(equalToConstant: 30).isActive = true
        resetBtn.addTarget(self, action: #selector(resetClassesTapped), for: .touchUpInside)
        chipsStack.addArrangedSubview(resetBtn)
    }

    @objc private func resetClassesTapped() {
        let alert = UIAlertController(title: "Reset Classes", message: "Restore default classes?", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Reset", style: .destructive) { [weak self] _ in
            guard let self = self else { return }
            self.resetToDefaultClasses()
            self.rebuildChips()
            self.updateStatus("Classes reset to defaults")
        })
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }

    private func makeChip(title: String, isActive: Bool) -> UIButton {
        let btn = UIButton(type: .system)
        btn.setTitle(title, for: .normal)
        btn.titleLabel?.font = .systemFont(ofSize: 12, weight: .medium)
        if isActive {
            btn.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.8)
            btn.setTitleColor(.white, for: .normal)
        } else {
            btn.backgroundColor = UIColor.darkGray.withAlphaComponent(0.6)
            btn.setTitleColor(.gray, for: .normal)
        }
        btn.layer.cornerRadius = 14
        btn.clipsToBounds = true
        btn.contentEdgeInsets = UIEdgeInsets(top: 4, left: 12, bottom: 4, right: 12)
        btn.addTarget(self, action: #selector(chipTapped(_:)), for: .touchUpInside)
        let longPress = UILongPressGestureRecognizer(target: self, action: #selector(chipLongPressed(_:)))
        btn.addGestureRecognizer(longPress)
        return btn
    }

    @objc private func chipTapped(_ sender: UIButton) {
        guard let title = sender.title(for: .normal) else { return }
        if activeClasses.contains(title) {
            activeClasses.remove(title)
        } else {
            activeClasses.insert(title)
        }
        saveClasses()
        rebuildChips()
    }

    @objc private func chipLongPressed(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began,
              let btn = gesture.view as? UIButton,
              let title = btn.title(for: .normal) else { return }

        let alert = UIAlertController(title: "Remove Class", message: "Delete '\(title)'?", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Delete", style: .destructive) { [weak self] _ in
            guard let self = self else { return }
            self.savedClasses.removeAll { $0 == title }
            self.activeClasses.remove(title)
            self.saveClasses()
            self.rebuildChips()
        })
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }

    @objc private func addNewClass() {
        let alert = UIAlertController(title: "Add Class", message: nil, preferredStyle: .alert)
        alert.addTextField { tf in
            tf.placeholder = "e.g. bottle"
            tf.autocapitalizationType = .none
            tf.autocorrectionType = .no
        }
        alert.addAction(UIAlertAction(title: "Add", style: .default) { [weak self] _ in
            guard let self = self,
                  let text = alert.textFields?.first?.text?.trimmingCharacters(in: .whitespaces),
                  !text.isEmpty else { return }
            if !self.savedClasses.contains(text) {
                self.savedClasses.append(text)
                self.activeClasses.insert(text)
                self.saveClasses()
                self.rebuildChips()
            }
        })
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        present(alert, animated: true)
    }

    // MARK: - Settings

    @objc private func showSettings() {
        let ds = DetectionService.shared
        let unitNames = ["Meters", "Feet", "Centimeters"]
        let nextUnit = unitNames[(unitMode + 1) % 3]
        let unitLabel = "Units: \(unitNames[unitMode]) → \(nextUnit)"
        let serverLabel = "Server: \(ds.serverName)"

        let alert = UIAlertController(title: "Settings", message: serverLabel, preferredStyle: .alert)

        alert.addTextField { tf in
            tf.text = String(ds.scoreThreshold)
            tf.placeholder = "Score threshold (0.0 - 1.0)"
            tf.keyboardType = .decimalPad
        }

        alert.addAction(UIAlertAction(title: "Save", style: .default) { _ in
            if let threshStr = alert.textFields?[0].text, let thresh = Float(threshStr) {
                ds.scoreThreshold = max(0, min(1, thresh))
            }
        })

        alert.addAction(UIAlertAction(title: "Switch to Modal", style: .default) { [weak self] _ in
            ds.apiUrl = DetectionService.modalUrl
            ds.warmUp()
            self?.updateStatus("Server: Modal")
            print("[Settings] Switched to Modal")
        })

        alert.addAction(UIAlertAction(title: "Switch to ngrok", style: .default) { [weak self] _ in
            ds.apiUrl = DetectionService.ngrokUrl
            self?.updateStatus("Server: ngrok")
            print("[Settings] Switched to ngrok")
        })

        alert.addAction(UIAlertAction(title: unitLabel, style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.unitMode = (self.unitMode + 1) % 3
            let names = ["Meters", "Feet", "Centimeters"]
            self.updateStatus("Units: \(names[self.unitMode])")
            print("[Settings] Units switched to \(names[self.unitMode])")
        })

        let localTitle = useLocalModel ? "Switch to Server" : "Switch to Local (ONNX)"
        alert.addAction(UIAlertAction(title: localTitle, style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.useLocalModel.toggle()
            let modeStr = self.useLocalModel ? "Local (ONNX)" : "Server (\(ds.serverName))"
            self.updateStatus("Mode: \(modeStr)")
            print("[Settings] Inference mode: \(modeStr)")
        })

        alert.addAction(UIAlertAction(title: "Clear Boxes", style: .destructive) { [weak self] _ in
            self?.bboxRenderer.clearBoxes()
            self?.updateStatus("Boxes cleared")
        })

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))

        present(alert, animated: true)
    }
}

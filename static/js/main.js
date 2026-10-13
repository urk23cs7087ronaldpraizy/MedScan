// MedScan AI – NeoMedical Interactive Engine
// Triage, Analytics, Comparison Slider, Radial Gauges, Keyboard Shortcuts

document.addEventListener('DOMContentLoaded', function () {
    // ==================== DOM REFERENCES ====================
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const uploadPreview = document.getElementById('uploadPreview');
    const uploadContent = document.querySelector('.upload-content');
    const uploadLoading = document.getElementById('uploadLoading');
    const removeImage = document.getElementById('removeImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const enhanceBtn = document.getElementById('enhanceBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const resultsSection = document.getElementById('results');
    const modalityBtns = document.querySelectorAll('.modality-btn');
    const enhanceBtns = document.querySelectorAll('.enhance-btn');
    const assistantInput = document.getElementById('assistantInput');
    const sendMessageBtn = document.getElementById('sendMessageBtn');
    const loadHistoryBtn = document.getElementById('loadHistoryBtn');

    let selectedModality = 'xray';
    let selectedEnhancement = 'auto';
    let selectedFile = null;
    let currentPrediction = null;
    let originalImageSrc = null; // for comparison slider
    let userLocation = null;     // { lat, lng }
    let hospitalMap = null;      // Leaflet map instance
    let hospitalMarkers = [];    // Leaflet markers

    // ==================== GEOLOCATION ====================
    function getUserLocation() {
        if (!navigator.geolocation) return;
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                console.log('📍 Location acquired:', userLocation);
            },
            (err) => {
                console.log('⚠ Geolocation denied:', err.message);
            },
            { enableHighAccuracy: true, timeout: 10000 }
        );
    }
    getUserLocation();

    // ==================== NAVBAR ====================
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);

        // Active nav link tracking
        const sections = document.querySelectorAll('section[id]');
        let current = '';
        sections.forEach(sec => {
            const top = sec.offsetTop - 120;
            if (window.scrollY >= top) current = sec.id;
        });
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === '#' + current);
        });
    });

    navToggle.addEventListener('click', () => navLinks.classList.toggle('active'));

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
                navLinks.classList.remove('active');
            }
        });
    });

    // ==================== SECTION ENTRANCE ANIMATIONS ====================
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                sectionObserver.unobserve(entry.target); // once only
            }
        });
    }, { threshold: 0.08 });

    document.querySelectorAll('.section-animate').forEach(el => sectionObserver.observe(el));

    // ==================== MODALITY & ENHANCEMENT SELECTION ====================
    modalityBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modalityBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModality = btn.dataset.modality;
        });
    });

    enhanceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            enhanceBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedEnhancement = btn.dataset.enhance;
        });
    });

    // ==================== FILE UPLOAD ====================
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) { showToast('Please select an image file', 'error'); return; }
        if (file.size > 16 * 1024 * 1024) { showToast('File size must be less than 16MB', 'error'); return; }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            originalImageSrc = e.target.result; // save for comparison
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        originalImageSrc = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadPreview.style.display = 'none';
        uploadContent.style.display = 'block';
        analyzeBtn.disabled = true;
        hideComparison();
    }

    // ==================== IMAGE ENHANCEMENT ====================
    enhanceBtn.addEventListener('click', async () => {
        if (!selectedFile) { showToast('Please select an image first', 'error'); return; }

        uploadPreview.style.display = 'none';
        uploadLoading.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('type', selectedEnhancement);

        try {
            const response = await fetch('/enhance', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.success) {
                previewImage.src = data.enhanced_image;
                uploadPreview.style.display = 'flex';
                showToast(`Image enhanced! Quality: ${data.quality.quality}`, 'success');

                // Show comparison slider
                showComparison(originalImageSrc, data.enhanced_image, data.quality);
            } else {
                uploadPreview.style.display = 'flex';
                showToast(data.error || 'Enhancement failed', 'error');
            }
        } catch (error) {
            uploadPreview.style.display = 'flex';
            showToast('Network error: ' + error.message, 'error');
        } finally {
            uploadLoading.style.display = 'none';
        }
    });

    // ==================== IMAGE COMPARISON SLIDER ====================
    const compContainer = document.getElementById('comparisonContainer');
    const compSlider = document.getElementById('comparisonSlider');
    const compOriginal = document.getElementById('compOriginal');
    const compEnhanced = document.getElementById('compEnhanced');
    const compDivider = document.getElementById('compDivider');
    const compHandle = document.getElementById('compHandle');
    const compMetrics = document.getElementById('compMetrics');
    const closeComparison = document.getElementById('closeComparison');

    function showComparison(origSrc, enhSrc, quality) {
        compOriginal.src = origSrc;
        compEnhanced.src = enhSrc;
        compContainer.classList.add('visible');

        // Reset position to 50%
        updateComparisonPosition(50);

        // Show quality metrics
        if (quality) {
            compMetrics.innerHTML = `
                <div class="metric-item"><div class="metric-value">${quality.brightness?.toFixed(0) || '—'}</div><div class="metric-label">Brightness</div></div>
                <div class="metric-item"><div class="metric-value">${quality.contrast?.toFixed(0) || '—'}</div><div class="metric-label">Contrast</div></div>
                <div class="metric-item"><div class="metric-value">${quality.sharpness?.toFixed(0) || '—'}</div><div class="metric-label">Sharpness</div></div>
                <div class="metric-item"><div class="metric-value">${quality.score || '—'}</div><div class="metric-label">Score</div></div>
            `;
        }
    }

    function hideComparison() {
        compContainer.classList.remove('visible');
    }

    closeComparison.addEventListener('click', hideComparison);

    function updateComparisonPosition(pct) {
        pct = Math.max(5, Math.min(95, pct));
        compEnhanced.style.clipPath = `inset(0 0 0 ${pct}%)`;
        compDivider.style.left = pct + '%';
        compHandle.style.left = pct + '%';
    }

    let isDraggingComp = false;
    compSlider.addEventListener('mousedown', () => isDraggingComp = true);
    compSlider.addEventListener('touchstart', () => isDraggingComp = true, { passive: true });
    window.addEventListener('mouseup', () => isDraggingComp = false);
    window.addEventListener('touchend', () => isDraggingComp = false);

    function handleCompMove(clientX) {
        if (!isDraggingComp) return;
        const rect = compSlider.getBoundingClientRect();
        const pct = ((clientX - rect.left) / rect.width) * 100;
        updateComparisonPosition(pct);
    }
    compSlider.addEventListener('mousemove', (e) => handleCompMove(e.clientX));
    compSlider.addEventListener('touchmove', (e) => handleCompMove(e.touches[0].clientX), { passive: true });
    // Click to position
    compSlider.addEventListener('click', (e) => {
        const rect = compSlider.getBoundingClientRect();
        updateComparisonPosition(((e.clientX - rect.left) / rect.width) * 100);
    });

    // ==================== ANALYZE ====================
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) { showToast('Please select an image first', 'error'); return; }

        uploadPreview.style.display = 'none';
        uploadLoading.style.display = 'flex';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('modality', selectedModality);
        formData.append('patient_name', document.getElementById('patientName').value || '');
        formData.append('patient_age', document.getElementById('patientAge').value || '');
        formData.append('patient_gender', document.getElementById('patientGender').value || '');

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.success) {
                if (data.prediction.error) {
                    showToast(data.prediction.message, 'error');
                    uploadLoading.style.display = 'none';
                    uploadPreview.style.display = 'flex';
                    analyzeBtn.disabled = false;
                    return;
                }
                currentPrediction = data.prediction;
                displayResults(data.prediction, data.hospitals, data.triage);
                showToast('Analysis complete!', 'success');
            } else {
                showToast(data.error || 'Analysis failed', 'error');
            }
        } catch (error) {
            showToast('Network error: ' + error.message, 'error');
        } finally {
            uploadLoading.style.display = 'none';
            uploadContent.style.display = 'block';
            analyzeBtn.disabled = false;
        }
    });

    // ==================== DISPLAY RESULTS ====================
    function displayResults(prediction, hospitals, triage) {
        if (!prediction.is_valid_medical_image) {
            showToast(prediction.message || 'Invalid medical image', 'error');
            return;
        }
        currentPrediction = prediction;

        // Disease name
        document.getElementById('diseaseName').textContent = prediction.prediction;

        // Severity
        const severityBadge = document.getElementById('severityBadge');
        const severity = prediction.severity || 'Normal';
        severityBadge.className = 'severity-badge severity-' + severity.toLowerCase().replace(' ', '-');
        severityBadge.innerHTML = `<span>${severity}</span>`;

        // Radial confidence gauge
        const conf = prediction.confidence;
        const circumference = 2 * Math.PI * 65;
        const gaugeFill = document.getElementById('gaugeFill');
        const offset = circumference - (conf / 100) * circumference;
        setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 100);
        animateCounter(document.getElementById('gaugeValue'), 0, conf, 1200, '%');

        // Badge
        let badgeClass, badgeText;
        if (conf >= 80) { badgeClass = 'high'; badgeText = 'High Confidence'; }
        else if (conf >= 50) { badgeClass = 'medium'; badgeText = 'Medium Confidence'; }
        else { badgeClass = 'low'; badgeText = 'Low Confidence'; }
        document.getElementById('predictionBadge').className = 'prediction-badge ' + badgeClass;
        document.getElementById('confidenceLevel').textContent = badgeText;

        // Image quality
        const quality = prediction.image_quality?.quality || 'Unknown';
        document.getElementById('imageQuality').innerHTML = `<i class="fa-solid fa-image"></i><span>Quality: ${quality}</span>`;

        // Triage dashboard
        if (triage) { displayTriage(triage); }

        // Differential diagnoses
        const differentialList = document.getElementById('differentialList');
        const allProbs = prediction.all_probabilities || {};
        const sortedProbs = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);

        if (sortedProbs.length > 1) {
            differentialList.innerHTML = sortedProbs.map(([disease, prob], index) => `
                <div class="differential-item" style="animation-delay: ${index * 0.1}s;">
                    <span class="differential-rank">${index + 1}</span>
                    <span class="differential-name">${disease}</span>
                    <span class="differential-prob">${prob.toFixed(1)}%</span>
                </div>
            `).join('');
        } else {
            differentialList.innerHTML = '<p class="no-differential">Single diagnosis available</p>';
        }

        // CHANGE 1: Render Chart.js visualizations
        renderPredictionCharts(allProbs, prediction);

        // Explanation
        document.getElementById('diseaseDescription').textContent = prediction.description;
        document.getElementById('causesList').innerHTML = (prediction.causes || []).map(c => `<li>${c}</li>`).join('');
        document.getElementById('symptomsList').innerHTML = (prediction.symptoms || []).map(s => `<li>${s}</li>`).join('');
        document.getElementById('riskFactorsList').innerHTML = (prediction.risk_factors || []).map(r => `<li>${r}</li>`).join('');
        document.getElementById('treatmentList').innerHTML = (prediction.treatment || []).map(t => `<li>${t}</li>`).join('');
        document.getElementById('preventionList').innerHTML = (prediction.prevention || []).map(p => `<li>${p}</li>`).join('');

        // Hospitals
        currentPrediction._static_hospitals = hospitals || [];
        loadNearbyHospitals(prediction.prediction);

        // Show results
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        setupAccordion();
    }

    // ==================== CHANGE 1: CHART.JS VISUALIZATION ====================
    let barChartInstance = null;

    function renderPredictionCharts(allProbs, prediction) {
        const labels = Object.keys(allProbs);
        const values = Object.values(allProbs);
        const chartColors = ['#06b6d4','#6366f1','#10b981','#fbbf24','#f43f5e','#818cf8'];

        // Risk Level Indicator
        const conf = prediction.confidence || 0;
        const sev = (prediction.severity || 'normal').toLowerCase();
        let riskLevel = 'LOW', riskColor = '#10b981', riskPct = 20;
        if (sev === 'critical' || (conf > 85 && sev !== 'normal')) { riskLevel = 'CRITICAL'; riskColor = '#ef4444'; riskPct = 95; }
        else if (sev === 'high' || sev === 'severe') { riskLevel = 'HIGH'; riskColor = '#f43f5e'; riskPct = 75; }
        else if (sev === 'medium' || sev === 'moderate') { riskLevel = 'MEDIUM'; riskColor = '#fbbf24'; riskPct = 50; }
        else if (sev === 'mild') { riskLevel = 'LOW'; riskColor = '#10b981'; riskPct = 25; }

        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = riskLevel;
        riskBadge.style.background = riskColor;
        const riskFill = document.getElementById('riskMeterFill');
        riskFill.style.background = riskColor;
        setTimeout(() => { riskFill.style.width = riskPct + '%'; }, 200);

        // Destroy old chart
        if (barChartInstance) barChartInstance.destroy();

        if (labels.length === 0) return;

        const chartDefaults = { color: '#94a3b8', font: { family: 'DM Sans' } };

        // Bar Chart
        barChartInstance = new Chart(document.getElementById('probBarChart'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{ label: 'Confidence %', data: values, backgroundColor: chartColors.slice(0, labels.length), borderRadius: 6, borderSkipped: false }]
            },
            options: {
                responsive: true, plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, max: 100, ticks: chartDefaults, grid: { color: 'rgba(99,102,241,0.08)' } },
                    x: { ticks: chartDefaults, grid: { display: false } }
                }
            }
        });
    }

    // ==================== NEARBY HOSPITALS SYSTEM ====================
    async function loadNearbyHospitals(diagnosis, radius) {
        const locationStatus = document.getElementById('locationStatus');
        const hospitalMapEl = document.getElementById('hospitalMap');
        const hospitalsList = document.getElementById('hospitalsList');
        const radiusSelector = document.getElementById('radiusSelector');
        const searchRadius = document.getElementById('searchRadius');

        radius = radius || parseInt(searchRadius?.value || '10000');

        // If we don't have location yet, try to get it
        if (!userLocation) {
            locationStatus.innerHTML = `
                <div class="location-loading">
                    <div class="loader" style="width:20px;height:20px;"></div>
                    <span>Requesting your location… Please allow location access.</span>
                </div>`;
            locationStatus.style.display = 'block';

            try {
                const pos = await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(resolve, reject, {
                        enableHighAccuracy: true, timeout: 10000
                    });
                });
                userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
            } catch (e) {
                locationStatus.innerHTML = `
                    <div class="location-error" style="display:flex; align-items:center; gap:10px; padding:14px; background:rgba(239,68,68,0.1); border-radius:10px; border:1px solid rgba(239,68,68,0.2);">
                        <i class="fa-solid fa-location-crosshairs" style="color:#ef4444; font-size:18px;"></i>
                        <div>
                            <div style="font-weight:600; color:#ef4444; margin-bottom:2px;">Location access denied</div>
                            <div style="font-size:12px; color:var(--text-muted);">Enable location in your browser settings to find hospitals near you. Showing general recommendations instead.</div>
                        </div>
                    </div>`;
                // Fallback to static hospital data
                displayStaticHospitals(diagnosis);
                return;
            }
        }

        // Show loading
        locationStatus.innerHTML = `
            <div class="location-loading">
                <div class="loader" style="width:20px;height:20px;"></div>
                <span>Finding hospitals near you…</span>
            </div>`;
        locationStatus.style.display = 'block';

        try {
            const resp = await fetch('/nearby-hospitals', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lat: userLocation.lat,
                    lng: userLocation.lng,
                    prediction: diagnosis || '',
                    radius: radius
                })
            });
            const data = await resp.json();

            if (!data.success || !data.hospitals || data.hospitals.length === 0) {
                locationStatus.innerHTML = `
                    <div class="location-error" style="display:flex; align-items:center; gap:10px; padding:14px; background:rgba(251,191,36,0.1); border-radius:10px; border:1px solid rgba(251,191,36,0.2);">
                        <i class="fa-solid fa-triangle-exclamation" style="color:#fbbf24; font-size:18px;"></i>
                        <div>
                            <div style="font-weight:600; color:#fbbf24; margin-bottom:2px;">No hospitals found nearby</div>
                            <div style="font-size:12px; color:var(--text-muted);">Try increasing the search radius. Showing general recommendations.</div>
                        </div>
                    </div>`;
                radiusSelector.style.display = 'flex';
                displayStaticHospitals(diagnosis);
                return;
            }

            // Success — show location found
            locationStatus.innerHTML = `
                <div style="display:flex; align-items:center; gap:10px; padding:10px 14px; background:rgba(16,185,129,0.1); border-radius:10px; border:1px solid rgba(16,185,129,0.2);">
                    <i class="fa-solid fa-location-dot" style="color:#10b981; font-size:16px;"></i>
                    <span style="font-size:13px; color:#10b981; font-weight:500;">
                        Found <strong>${data.count}</strong> hospitals within ${radius / 1000} km of your location
                        ${data.total_found > data.count ? ` (${data.total_found} total)` : ''}
                    </span>
                </div>`;
            radiusSelector.style.display = 'flex';

            // Render map
            renderHospitalMap(data.hospitals, data.user_lat, data.user_lng);

            // Render hospital cards
            renderHospitalCards(data.hospitals);

        } catch (err) {
            console.error('Hospital fetch error:', err);
            locationStatus.innerHTML = `
                <div class="location-error" style="padding:12px; background:rgba(239,68,68,0.1); border-radius:10px; color:#ef4444; font-size:13px;">
                    <i class="fa-solid fa-circle-xmark"></i> Could not fetch nearby hospitals. Showing general recommendations.
                </div>`;
            displayStaticHospitals(diagnosis);
        }
    }

    function renderHospitalMap(hospitals, userLat, userLng) {
        const mapEl = document.getElementById('hospitalMap');
        mapEl.style.display = 'block';

        // Destroy previous map
        if (hospitalMap) {
            hospitalMap.remove();
            hospitalMap = null;
        }

        // Initialize Leaflet map
        hospitalMap = L.map('hospitalMap', {
            zoomControl: true,
            scrollWheelZoom: true
        }).setView([userLat, userLng], 13);

        // Dark tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap © CARTO',
            maxZoom: 19
        }).addTo(hospitalMap);

        // User location marker
        const userIcon = L.divIcon({
            className: 'user-marker',
            html: '<div style="width:16px;height:16px;background:#06b6d4;border:3px solid #fff;border-radius:50%;box-shadow:0 0 12px rgba(6,182,212,0.6);"></div>',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        });
        L.marker([userLat, userLng], { icon: userIcon })
            .addTo(hospitalMap)
            .bindPopup('<strong style="color:#06b6d4;">📍 Your Location</strong>');

        // Clear previous markers
        hospitalMarkers = [];

        // Hospital markers
        const bounds = L.latLngBounds([[userLat, userLng]]);

        hospitals.forEach((h, i) => {
            const isEmergency = h.emergency;
            const markerColor = isEmergency ? '#ef4444' : '#fbbf24';
            const hospIcon = L.divIcon({
                className: 'hospital-marker',
                html: `<div style="width:30px;height:30px;background:${markerColor};border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);border:2px solid #fff;font-size:13px;">
                    <span style="color:#fff;font-weight:700;">${i + 1}</span>
                </div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            });

            const marker = L.marker([h.lat, h.lng], { icon: hospIcon })
                .addTo(hospitalMap)
                .bindPopup(`
                    <div style="min-width:200px;">
                        <strong style="font-size:14px;">${h.name}</strong><br>
                        <span style="font-size:12px; color:#666;">${h.address}</span><br>
                        <span style="font-size:12px; color:#06b6d4;">📏 ${h.distance}</span>
                        ${h.emergency ? '<br><span style="color:#ef4444; font-size:12px; font-weight:600;">🚑 Emergency Available</span>' : ''}
                        <br><a href="${h.directions_url}" target="_blank" style="color:#6366f1; font-size:12px; font-weight:600; text-decoration:none;">
                            🗺️ Get Directions →
                        </a>
                    </div>
                `);

            hospitalMarkers.push(marker);
            bounds.extend([h.lat, h.lng]);
        });

        // Fit map to show all markers
        if (hospitals.length > 0) {
            hospitalMap.fitBounds(bounds, { padding: [30, 30] });
        }

        // Fix map render on hidden containers
        setTimeout(() => { hospitalMap.invalidateSize(); }, 300);
    }

    function renderHospitalCards(hospitals) {
        const hospitalsList = document.getElementById('hospitalsList');
        hospitalsList.innerHTML = hospitals.map((h, i) => `
            <div class="hospital-card" data-index="${i}"
                 onmouseenter="document.dispatchEvent(new CustomEvent('hospital-hover', {detail:${i}}))"
                 onmouseleave="document.dispatchEvent(new CustomEvent('hospital-unhover', {detail:${i}}))">
                <div class="hospital-rank">${i + 1}</div>
                <div class="hospital-info">
                    <div class="hospital-name">
                        ${h.name}
                        ${h.emergency ? '<span class="emergency-badge"><i class="fa-solid fa-truck-medical"></i> ER</span>' : ''}
                    </div>
                    <div class="hospital-address"><i class="fa-solid fa-map-pin" style="margin-right:4px;color:#6366f1;"></i>${h.address}</div>
                    <div class="hospital-meta">
                        <span class="hospital-distance">
                            <i class="fa-solid fa-route"></i>
                            ${h.distance}
                        </span>
                        <div class="hospital-specialties">
                            ${h.specialties.slice(0, 3).map(s => `<span class="specialty-badge">${s}</span>`).join('')}
                        </div>
                    </div>
                    ${h.opening_hours ? `<div style="font-size:11px; color:var(--text-muted); margin-top:4px;"><i class="fa-solid fa-clock" style="margin-right:4px;"></i>${h.opening_hours}</div>` : ''}
                    <div class="hospital-actions">
                        <a href="${h.directions_url}" target="_blank" class="hospital-action-btn directions">
                            <i class="fa-solid fa-diamond-turn-right"></i> Get Directions
                        </a>
                        ${h.phone ? `<a href="tel:${h.phone}" class="hospital-action-btn call-btn"><i class="fa-solid fa-phone"></i> ${h.phone}</a>` : ''}
                        ${h.website ? `<a href="${h.website}" target="_blank" class="hospital-action-btn" style="background:rgba(99,102,241,0.15); color:#818cf8;"><i class="fa-solid fa-globe"></i> Website</a>` : ''}
                        <a href="${h.maps_url}" target="_blank" class="hospital-action-btn" style="background:rgba(16,185,129,0.15); color:#10b981;">
                            <i class="fa-solid fa-map-location-dot"></i> View on Map
                        </a>
                    </div>
                </div>
            </div>
        `).join('');
    }

    // Highlight marker on hospital card hover
    document.addEventListener('hospital-hover', (e) => {
        const idx = e.detail;
        if (hospitalMarkers[idx]) {
            hospitalMarkers[idx].openPopup();
        }
    });
    document.addEventListener('hospital-unhover', (e) => {
        const idx = e.detail;
        if (hospitalMarkers[idx]) {
            hospitalMarkers[idx].closePopup();
        }
    });

    function displayStaticHospitals(diagnosis) {
        // Fallback: show the pre-configured hospital data from the /predict response
        if (currentPrediction && currentPrediction._static_hospitals) {
            const hospitals = currentPrediction._static_hospitals;
            const hospitalsList = document.getElementById('hospitalsList');
            hospitalsList.innerHTML = hospitals.map(hospital => `
                <div class="hospital-card">
                    <div class="hospital-image">
                        <i class="fa-solid fa-hospital"></i>
                    </div>
                    <div class="hospital-info">
                        <div class="hospital-name">${hospital.name}</div>
                        <div class="hospital-address">${hospital.address}</div>
                        <div class="hospital-meta">
                            <span class="hospital-rating">
                                <i class="fa-solid fa-star"></i>
                                ${hospital.rating}
                            </span>
                            <div class="hospital-specialties">
                                ${hospital.specialties.slice(0, 2).map(s => `<span class="specialty-badge">${s}</span>`).join('')}
                            </div>
                        </div>
                        <div class="hospital-actions">
                            <a href="https://maps.google.com/?q=${encodeURIComponent(hospital.name + ' ' + hospital.address)}" target="_blank" class="hospital-action-btn directions">
                                <i class="fa-solid fa-diamond-turn-right"></i> Directions
                            </a>
                            ${hospital.phone ? `<a href="tel:${hospital.phone}" class="hospital-action-btn call-btn"><i class="fa-solid fa-phone"></i> Call</a>` : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        }
    }

    // Refresh button
    document.getElementById('refreshHospitalsBtn')?.addEventListener('click', () => {
        if (currentPrediction) {
            loadNearbyHospitals(currentPrediction.prediction);
        }
    });

    // Radius change — live update label + refresh
    document.getElementById('searchRadius')?.addEventListener('change', (e) => {
        const radiusM = parseInt(e.target.value);
        const radiusKm = radiusM / 1000;
        const label = document.getElementById('radiusLabel');
        if (label) label.textContent = `${radiusKm} km radius`;
        if (currentPrediction) {
            loadNearbyHospitals(currentPrediction.prediction, radiusM);
        }
    });

    // Also update label on initial show
    function updateRadiusLabel() {
        const sel = document.getElementById('searchRadius');
        const label = document.getElementById('radiusLabel');
        if (sel && label) label.textContent = `${parseInt(sel.value) / 1000} km radius`;
    }

    // ==================== TRIAGE DASHBOARD ====================
    function displayTriage(triage) {
        const triageCard = document.getElementById('triageCard');
        triageCard.style.display = 'block';

        const score = triage.urgency_score || 0;
        const level = triage.urgency_level || 'low';
        const notes = triage.clinical_notes || [];

        // Urgency ring
        const circumference = 2 * Math.PI * 70; // r=70
        const urgencyFill = document.getElementById('urgencyFill');
        const offset = circumference - (score / 100) * circumference;

        // Set gradient based on level
        const gradientMap = { low: 'urgencyGradientLow', moderate: 'urgencyGradientMod', high: 'urgencyGradientHigh', critical: 'urgencyGradientCrit' };
        urgencyFill.setAttribute('stroke', `url(#${gradientMap[level] || gradientMap.low})`);
        setTimeout(() => { urgencyFill.style.strokeDashoffset = offset; }, 200);

        // Animate score counter
        animateCounter(document.getElementById('urgencyScore'), 0, score, 1500);

        // Badge
        const urgencyBadge = document.getElementById('urgencyBadge');
        urgencyBadge.textContent = level.toUpperCase();
        urgencyBadge.className = 'urgency-level-badge urgency-' + level;

        // Clinical notes
        const clinicalNotes = document.getElementById('clinicalNotes');
        clinicalNotes.innerHTML = notes.map(note => `
            <div class="clinical-note">
                <i class="fa-solid fa-circle-info"></i>
                <span>${note}</span>
            </div>
        `).join('');

        // Timestamp
        if (triage.generated_at) {
            const d = new Date(triage.generated_at);
            document.getElementById('triageTimestamp').textContent = `Generated: ${d.toLocaleString()}`;
        }
    }

    // ==================== ANIMATED COUNTER ====================
    function animateCounter(el, start, end, duration, suffix = '') {
        const startTime = performance.now();
        const isFloat = !Number.isInteger(end);
        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * eased;
            el.textContent = (isFloat ? current.toFixed(1) : Math.round(current)) + suffix;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    // ==================== ACCORDION ====================
    function setupAccordion() {
        document.querySelectorAll('.accordion-item').forEach(item => {
            const header = item.querySelector('.accordion-header');
            // Remove old listeners by cloning
            const newHeader = header.cloneNode(true);
            header.parentNode.replaceChild(newHeader, header);
            newHeader.addEventListener('click', () => {
                const isActive = item.classList.contains('active');
                item.parentElement.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('active'));
                if (!isActive) item.classList.add('active');
                newHeader.setAttribute('aria-expanded', !isActive);
            });
        });
    }

    // ==================== ANALYTICS DASHBOARD ====================
    async function loadAnalytics() {
        try {
            const response = await fetch('/statistics');
            const data = await response.json();
            if (!data.success) return;

            const stats = data.statistics;

            // Animated counters
            animateCounter(document.getElementById('statTotal'), 0, stats.total || 0, 1000);
            animateCounter(document.getElementById('statConfidence'), 0, stats.avg_confidence || 0, 1000, '%');
            animateCounter(document.getElementById('statValid'), 0, stats.valid_images || 0, 1000);

            const conditionCount = Object.keys(stats.by_prediction || {}).length;
            animateCounter(document.getElementById('statConditions'), 0, conditionCount, 1000);

            // Donut chart
            renderDonutChart(stats.by_modality || {}, stats.total || 0);

            // Bar chart
            renderBarChart(stats.by_prediction || {});

        } catch (error) {
            console.log('Analytics load error:', error);
        }
    }

    function renderDonutChart(byModality, total) {
        const donutChart = document.getElementById('donutChart');
        const donutLegend = document.getElementById('donutLegend');
        document.getElementById('donutTotal').textContent = total;

        const colors = {
            xray: '#06b6d4',
            mri: '#6366f1',
            ct: '#f43f5e'
        };
        const labels = { xray: 'X-Ray', mri: 'MRI', ct: 'CT Scan' };

        // Build conic gradient
        let segments = [];
        let runningPct = 0;
        const entries = Object.entries(byModality);

        entries.forEach(([modality, count]) => {
            const pct = total > 0 ? (count / total) * 100 : 0;
            const color = colors[modality] || '#64748b';
            segments.push(`${color} ${runningPct}% ${runningPct + pct}%`);
            runningPct += pct;
        });

        if (segments.length > 0) {
            // Fill remaining with bg
            if (runningPct < 100) segments.push(`#1a2340 ${runningPct}% 100%`);
            donutChart.style.background = `conic-gradient(${segments.join(', ')})`;
        } else {
            donutChart.style.background = '#1a2340';
        }

        // Legend
        donutLegend.innerHTML = entries.map(([modality, count]) => `
            <div class="legend-item">
                <span class="legend-dot" style="background: ${colors[modality] || '#64748b'}"></span>
                <span>${labels[modality] || modality}</span>
                <span class="legend-value">${count}</span>
            </div>
        `).join('') || '<div style="color: var(--text-muted); font-size: 13px;">No scans yet</div>';
    }

    function renderBarChart(byPrediction) {
        const barChart = document.getElementById('barChart');
        const entries = Object.entries(byPrediction).sort((a, b) => b[1] - a[1]).slice(0, 6);

        if (entries.length === 0) {
            barChart.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 40px;">
                <i class="fa-solid fa-chart-simple" style="font-size: 36px; margin-bottom: 12px; display: block;"></i>
                No data yet — analyze images to see trends
            </div>`;
            return;
        }

        const maxVal = Math.max(...entries.map(e => e[1]));
        const colors = ['#06b6d4', '#6366f1', '#10b981', '#fbbf24', '#f43f5e', '#818cf8'];

        barChart.innerHTML = entries.map(([condition, count], i) => {
            const pct = maxVal > 0 ? (count / maxVal) * 100 : 0;
            return `
                <div class="bar-item">
                    <span class="bar-label">${condition}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: 0%; background: ${colors[i % colors.length]};" data-width="${pct}%">${count}</div>
                    </div>
                </div>
            `;
        }).join('');

        // Animate bars
        setTimeout(() => {
            barChart.querySelectorAll('.bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 200);
    }

    // Load analytics on page load
    loadAnalytics();

    // ==================== DOWNLOAD REPORT (CHANGE 2: Fixed) ====================
    document.getElementById('downloadReportBtn').addEventListener('click', async () => {
        if (!currentPrediction) { showToast('No prediction to generate report', 'error'); return; }

        try {
            const response = await fetch('/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_name: document.getElementById('patientName').value,
                    patient_age: document.getElementById('patientAge').value,
                    patient_gender: document.getElementById('patientGender').value,
                    modality: currentPrediction.modality,
                    prediction: currentPrediction.prediction,
                    confidence: currentPrediction.confidence,
                    severity: currentPrediction.severity,
                    description: currentPrediction.description,
                    causes: currentPrediction.causes,
                    symptoms: currentPrediction.symptoms,
                    risk_factors: currentPrediction.risk_factors,
                    treatment: currentPrediction.treatment,
                    prevention: currentPrediction.prevention,
                    created_at: new Date().toISOString()
                })
            });
            const data = await response.json();
            if (data.success) {
                // CHANGE 2: Use location.href for proper download
                window.location.href = data.report_url;
                showToast('Report downloaded!', 'success');
            } else {
                showToast(data.error || 'Failed to generate report', 'error');
            }
        } catch (error) {
            showToast('Error: ' + error.message, 'error');
        }
    });

    // ==================== CONSULTATION MODAL ====================
    const consultationModal = document.getElementById('consultationModal');
    document.getElementById('consultationBtn').addEventListener('click', () => consultationModal.style.display = 'flex');
    document.getElementById('closeConsultationModal').addEventListener('click', () => consultationModal.style.display = 'none');

    document.getElementById('consultationForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('/consultation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_name: document.getElementById('consName').value,
                    email: document.getElementById('consEmail').value,
                    phone: document.getElementById('consPhone').value,
                    prediction_id: currentPrediction?.prediction_id,
                    modality: currentPrediction?.modality,
                    current_diagnosis: currentPrediction?.prediction,
                    symptoms_description: document.getElementById('consSymptoms').value,
                    additional_notes: document.getElementById('consNotes').value
                })
            });
            const data = await response.json();
            if (data.success) {
                // Show a nice confirmation with reference number
                const statusDiv = document.getElementById('consultationStatus');
                if (statusDiv) {
                    statusDiv.style.display = 'block';
                    statusDiv.innerHTML = `
                        <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:16px;text-align:center;">
                            <i class="fa-solid fa-circle-check" style="color:#10b981;font-size:28px;margin-bottom:8px;"></i>
                            <div style="color:#10b981;font-weight:600;margin-bottom:4px;">Request Submitted Successfully!</div>
                            <div style="color:#94a3b8;font-size:13px;">${data.message}</div>
                        </div>`;
                    setTimeout(() => { consultationModal.style.display = 'none'; statusDiv.style.display = 'none'; }, 5000);
                } else {
                    showToast(data.message || 'Consultation request submitted!', 'success');
                    consultationModal.style.display = 'none';
                }
                document.getElementById('consultationForm').reset();
            } else {
                showToast(data.error || 'Failed to submit', 'error');
            }
        } catch (error) {
            showToast('Error: ' + error.message, 'error');
        }
    });

    // ==================== FEEDBACK MODAL ====================
    const feedbackModal = document.getElementById('feedbackModal');
    document.getElementById('feedbackBtn').addEventListener('click', () => feedbackModal.style.display = 'flex');
    document.getElementById('closeFeedbackModal').addEventListener('click', () => feedbackModal.style.display = 'none');

    document.getElementById('feedbackForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentPrediction?.prediction_id) { showToast('No prediction to submit feedback for', 'error'); return; }

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: currentPrediction.prediction_id,
                    feedback_type: document.getElementById('feedbackType').value,
                    rating: document.getElementById('feedbackRating').value,
                    comment: document.getElementById('feedbackComment').value,
                    user_name: document.getElementById('feedbackName')?.value || 'Anonymous'
                })
            });
            const data = await response.json();
            if (data.success) {
                showToast('Thank you for your feedback!', 'success');
                feedbackModal.style.display = 'none';
                document.getElementById('feedbackForm').reset();
            } else {
                showToast(data.error || 'Failed to submit', 'error');
            }
        } catch (error) {
            showToast('Error: ' + error.message, 'error');
        }
    });

    // Close modals on outside click
    window.addEventListener('click', (e) => {
        if (e.target === consultationModal) consultationModal.style.display = 'none';
        if (e.target === feedbackModal) feedbackModal.style.display = 'none';
    });

    // ==================== AI ASSISTANT ====================
    function sendMessage() {
        const message = assistantInput.value.trim();
        if (!message) return;

        const messagesContainer = document.getElementById('assistantMessages');
        messagesContainer.innerHTML += `
            <div class="assistant-message user">
                <div class="message-content"><p>${escapeHtml(message)}</p></div>
                <div class="message-icon"><i class="fa-solid fa-user"></i></div>
            </div>
        `;
        assistantInput.value = '';
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        fetch('/assistant', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                messagesContainer.innerHTML += `
                    <div class="assistant-message bot">
                        <div class="message-icon"><i class="fa-solid fa-robot"></i></div>
                        <div class="message-content">${renderMarkdown(data.response)}</div>
                    </div>
                `;
            } else {
                messagesContainer.innerHTML += `
                    <div class="assistant-message bot">
                        <div class="message-icon"><i class="fa-solid fa-robot"></i></div>
                        <div class="message-content" style="color:#f43f5e;">Sorry, I couldn't process your request. Please try again.</div>
                    </div>
                `;
            }
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        })
        .catch(err => showToast('Assistant error: ' + err.message, 'error'));
    }

    function renderMarkdown(text) {
        // Convert **bold** to <strong>
        let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // Convert _italic_ to <em>
        html = html.replace(/\b_(.+?)_\b/g, '<em>$1</em>');
        // Convert ## headings
        html = html.replace(/^## (.+)$/gm, '<h4 style="color:#818cf8;margin:8px 0 4px;">$1</h4>');
        // Convert # headings
        html = html.replace(/^# (.+)$/gm, '<h3 style="color:#06b6d4;margin:8px 0 4px;">$1</h3>');
        // Bullet points — lines starting with • or -
        html = html.replace(/^[\u2022\-] (.+)$/gm, '<li style="margin:2px 0;">$1</li>');
        // Wrap consecutive <li> in a <ul>
        html = html.replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ul style="padding-left:18px;margin:6px 0;">$&</ul>');
        // Newlines to <br>
        html = html.replace(/\n/g, '<br>');
        // Clean double <br> after block elements
        html = html.replace(/(<\/ul>|<\/h[3-4]>|<\/p>)<br>/g, '$1');
        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    sendMessageBtn.addEventListener('click', sendMessage);
    assistantInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

    // ==================== HISTORY (CHANGE 6: Enhanced) ====================
    loadHistoryBtn.addEventListener('click', async () => {
        const modality = document.getElementById('historyModality').value;
        const search = document.getElementById('historySearch')?.value || '';
        try {
            const response = await fetch(`/history?modality=${modality || ''}&search=${encodeURIComponent(search)}`);
            const data = await response.json();
            if (data.success) displayHistory(data.predictions);
        } catch (error) {
            showToast('Error loading history: ' + error.message, 'error');
        }
    });

    function displayHistory(predictions) {
        const historyList = document.getElementById('historyList');
        if (!predictions || predictions.length === 0) {
            historyList.innerHTML = `<div class="history-empty"><i class="fa-solid fa-clock-rotate-left"></i><p>No prediction history yet</p></div>`;
            return;
        }
        historyList.innerHTML = predictions.map(p => {
            const date = new Date(p.created_at);
            const dateStr = date.toLocaleDateString('en-US', { year:'numeric', month:'short', day:'numeric' });
            const timeStr = date.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit' });
            const sev = (p.severity || 'normal').toLowerCase();
            const confPct = (p.confidence || 0).toFixed(1);
            return `
            <div class="history-detail-card">
                <div class="hdc-header">
                    <div class="hdc-modality">${(p.modality || '').toUpperCase()}</div>
                    <div class="hdc-date"><i class="fa-solid fa-calendar"></i> ${dateStr} <span style="color:var(--text-muted);margin-left:6px;">${timeStr}</span></div>
                </div>
                <div class="hdc-body">
                    <div class="hdc-prediction">${p.prediction}</div>
                    <div class="hdc-severity severity-${sev}">${p.severity || 'N/A'}</div>
                </div>
                <div class="hdc-meta">
                    ${p.patient_name ? `<span><i class="fa-solid fa-user"></i> ${p.patient_name}</span>` : ''}
                    ${p.patient_age ? `<span><i class="fa-solid fa-calendar-days"></i> Age ${p.patient_age}</span>` : ''}
                    ${p.patient_gender ? `<span><i class="fa-solid fa-venus-mars"></i> ${p.patient_gender}</span>` : ''}
                </div>
                <div class="hdc-confidence">
                    <div class="hdc-conf-bar"><div class="hdc-conf-fill" style="width:${confPct}%"></div></div>
                    <span>${confPct}%</span>
                </div>
            </div>`;
        }).join('');
    }

    // ==================== NEW ANALYSIS ====================
    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        document.getElementById('triageCard').style.display = 'none';

        // Reset gauge
        document.getElementById('gaugeFill').style.strokeDashoffset = 408;
        document.getElementById('gaugeValue').textContent = '0%';
        document.getElementById('urgencyFill').style.strokeDashoffset = 440;
        document.getElementById('urgencyScore').textContent = '0';
        document.getElementById('diseaseName').textContent = '-';

        resetUpload();
        document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });

        // Reload analytics
        loadAnalytics();
    });

    // ==================== KEYBOARD SHORTCUTS ====================
    const shortcutsOverlay = document.getElementById('shortcutsOverlay');

    document.addEventListener('keydown', (e) => {
        // Don't trigger when typing in inputs
        const active = document.activeElement;
        const isInput = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.tagName === 'SELECT');

        if (e.key === 'Escape') {
            // Close everything
            consultationModal.style.display = 'none';
            feedbackModal.style.display = 'none';
            shortcutsOverlay.classList.remove('visible');
            return;
        }

        if (isInput) return;

        if (e.key === '?') {
            e.preventDefault();
            shortcutsOverlay.classList.toggle('visible');
        }

        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'u' || e.key === 'U') {
                e.preventDefault();
                document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
                setTimeout(() => fileInput.click(), 400);
            }
            if (e.key === 'Enter') {
                e.preventDefault();
                if (!analyzeBtn.disabled) analyzeBtn.click();
            }
            if (e.key === 'k' || e.key === 'K') {
                e.preventDefault();
                document.getElementById('assistant').scrollIntoView({ behavior: 'smooth' });
                setTimeout(() => assistantInput.focus(), 400);
            }
        }
    });

    shortcutsOverlay.addEventListener('click', (e) => {
        if (e.target === shortcutsOverlay) shortcutsOverlay.classList.remove('visible');
    });

    // ==================== TOAST ====================
    function showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        const toastMessage = document.getElementById('toastMessage');
        toastMessage.textContent = message;
        toast.className = 'toast ' + type;

        const iconMap = {
            success: 'fa-solid fa-circle-check',
            error: 'fa-solid fa-circle-xmark',
            warning: 'fa-solid fa-triangle-exclamation',
            info: 'fa-solid fa-circle-info'
        };
        toast.querySelector('i').className = iconMap[type] || iconMap.success;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 3500);
    }

    // ==================== PARTICLES ====================
    createParticles();

    function createParticles() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;
        for (let i = 0; i < 25; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (12 + Math.random() * 12) + 's';
            particle.style.width = (2 + Math.random() * 3) + 'px';
            particle.style.height = particle.style.width;
            particlesContainer.appendChild(particle);
        }
    }

    // ==================== REAL-TIME COLLABORATION ====================
    let socket = null;
    let collabSessionId = null;
    let collabUsername = null;
    let chatMsgCount = 0;

    const collabJoinBtn = document.getElementById('collabJoinBtn');
    const collabLeaveBtn = document.getElementById('collabLeaveBtn');
    const collabChatInput = document.getElementById('collabChatInput');
    const collabSendBtn = document.getElementById('collabSendBtn');
    const collabMessages = document.getElementById('collabMessages');
    const collabIndicator = document.getElementById('collabIndicator');
    const collabStatusText = document.getElementById('collabStatusText');
    const collabSessionInfo = document.getElementById('collabSessionInfo');
    const collabCurrentSession = document.getElementById('collabCurrentSession');
    const onlineCount = document.getElementById('onlineCount');
    const chatBadge = document.getElementById('chatBadge');
    const copySessionBtn = document.getElementById('copySessionBtn');

    function initSocket() {
        if (socket) return;
        try {
            socket = io();

            socket.on('connect', () => {
                console.log('SocketIO connected');
            });

            socket.on('disconnect', () => {
                setCollabOffline();
            });

            socket.on('user_joined', (data) => {
                addSystemMessage(`${data.username} joined the session`);
            });

            socket.on('user_left', (data) => {
                addSystemMessage(`${data.username} left the session`);
            });

            socket.on('session_data', (data) => {
                // Load existing chat history
                if (data.chat && data.chat.length > 0) {
                    data.chat.forEach(msg => addChatMessage(msg, false));
                }
                // Update user count
                if (data.users) {
                    onlineCount.textContent = data.users.length;
                }
            });

            socket.on('status', (data) => {
                if (data.users_online !== undefined) {
                    onlineCount.textContent = data.users_online;
                }
                showToast(data.message, 'info');
            });

            socket.on('new_message', (data) => {
                addChatMessage(data, data.username !== collabUsername);
            });

        } catch (err) {
            console.error('SocketIO init error:', err);
            showToast('Could not connect to collaboration server', 'error');
        }
    }

    collabJoinBtn.addEventListener('click', async () => {
        collabUsername = document.getElementById('collabUsername').value.trim() || 'Doctor';
        const sessionInput = document.getElementById('collabSessionId').value.trim();
        const modality = document.getElementById('collabModality').value;

        initSocket();

        if (sessionInput) {
            // Join existing session
            collabSessionId = sessionInput;
        } else {
            // Create new session via REST, then join via socket
            try {
                const res = await fetch('/collab/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ modality })
                });
                const data = await res.json();
                if (data.success) {
                    collabSessionId = data.session_id;
                } else {
                    showToast('Failed to create session', 'error');
                    return;
                }
            } catch (err) {
                showToast('Network error: ' + err.message, 'error');
                return;
            }
        }

        // Join via socket
        socket.emit('join_collab', {
            session_id: collabSessionId,
            username: collabUsername,
            modality: document.getElementById('collabModality').value
        });

        setCollabOnline();
    });

    collabLeaveBtn.addEventListener('click', () => {
        if (socket) {
            socket.emit('leave_collab');
        }
        setCollabOffline();
    });

    function setCollabOnline() {
        collabIndicator.className = 'status-indicator online';
        collabStatusText.textContent = 'Connected';
        collabSessionInfo.style.display = 'flex';
        collabCurrentSession.textContent = collabSessionId;
        collabJoinBtn.style.display = 'none';
        collabLeaveBtn.style.display = 'inline-flex';
        collabChatInput.disabled = false;
        collabSendBtn.disabled = false;

        // Clear empty state
        const chatEmpty = collabMessages.querySelector('.chat-empty');
        if (chatEmpty) chatEmpty.remove();
    }

    function setCollabOffline() {
        collabIndicator.className = 'status-indicator offline';
        collabStatusText.textContent = 'Disconnected';
        collabSessionInfo.style.display = 'none';
        collabJoinBtn.style.display = 'inline-flex';
        collabLeaveBtn.style.display = 'none';
        collabChatInput.disabled = true;
        collabSendBtn.disabled = true;
        collabSessionId = null;
        chatMsgCount = 0;
        chatBadge.textContent = '0';
    }

    function addChatMessage(msg, isOther = true) {
        const isSelf = msg.username === collabUsername;
        const initials = (msg.username || '?').substring(0, 2);
        const time = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

        const div = document.createElement('div');
        div.className = `chat-msg${isSelf ? ' self' : ''}`;
        div.innerHTML = `
            <div class="chat-msg-avatar">${initials}</div>
            <div class="chat-msg-body">
                <div class="chat-msg-name">${escapeHtml(msg.username)}</div>
                <div>${escapeHtml(msg.message)}</div>
                <div class="chat-msg-time">${time}</div>
            </div>
        `;
        collabMessages.appendChild(div);
        collabMessages.scrollTop = collabMessages.scrollHeight;

        chatMsgCount++;
        chatBadge.textContent = chatMsgCount;
    }

    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'chat-system-msg';
        div.textContent = text;
        collabMessages.appendChild(div);
        collabMessages.scrollTop = collabMessages.scrollHeight;
    }

    function sendCollabMessage() {
        const msg = collabChatInput.value.trim();
        if (!msg || !socket || !collabSessionId) return;

        socket.emit('chat_message', { message: msg });

        // Show locally immediately
        addChatMessage({
            username: collabUsername,
            message: msg,
            timestamp: new Date().toISOString()
        }, false);

        collabChatInput.value = '';
    }

    collabSendBtn.addEventListener('click', sendCollabMessage);
    collabChatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendCollabMessage();
    });

    copySessionBtn.addEventListener('click', () => {
        if (collabSessionId) {
            navigator.clipboard.writeText(collabSessionId).then(() => {
                showToast('Session ID copied!', 'success');
            }).catch(() => {
                showToast('Could not copy', 'error');
            });
        }
    });

    // ==================== VOICE ASSISTANT (Web Speech API) ====================
    const micButton = document.getElementById('micButton');
    const micIcon = document.getElementById('micIcon');
    const micRing = document.getElementById('micRing');
    const voiceStatus = document.getElementById('voiceStatus');
    const voiceWave = document.getElementById('voiceWave');
    const voiceTranscript = document.getElementById('voiceTranscript');
    const voiceResponse = document.getElementById('voiceResponse');
    const speakResponseBtn = document.getElementById('speakResponseBtn');
    const promptChips = document.querySelectorAll('.prompt-chip');

    let isListening = false;
    let recognition = null;
    let lastVoiceResponse = '';

    // Check browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            micRing.classList.add('listening');
            micIcon.className = 'fa-solid fa-stop';
            voiceStatus.textContent = 'Listening...';
            voiceStatus.classList.add('active');
            voiceWave.classList.add('active');
            voiceTranscript.textContent = '';
            voiceTranscript.classList.add('active');
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            voiceTranscript.textContent = finalTranscript || interimTranscript;

            if (finalTranscript) {
                processVoiceQuery(finalTranscript);
            }
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopListening();
            if (event.error === 'not-allowed') {
                voiceStatus.textContent = 'Microphone access denied. Please allow mic access.';
                showToast('Microphone permission required', 'error');
            } else if (event.error === 'no-speech') {
                voiceStatus.textContent = 'No speech detected. Try again.';
            } else {
                voiceStatus.textContent = `Error: ${event.error}. Try again.`;
            }
        };

        recognition.onend = () => {
            stopListening();
        };
    }

    micButton.addEventListener('click', () => {
        if (!SpeechRecognition) {
            showToast('Your browser does not support speech recognition. Try Chrome.', 'warning');
            return;
        }

        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });

    function stopListening() {
        isListening = false;
        micRing.classList.remove('listening');
        micIcon.className = 'fa-solid fa-microphone';
        voiceWave.classList.remove('active');
        if (!voiceTranscript.textContent || voiceTranscript.textContent === '') {
            voiceStatus.textContent = 'Tap the microphone to speak';
            voiceStatus.classList.remove('active');
        }
    }

    async function processVoiceQuery(query) {
        voiceStatus.textContent = 'Processing your question...';
        voiceResponse.textContent = 'Thinking...';
        voiceResponse.classList.add('active');

        try {
            // CHANGE 5: Use context-aware voice endpoint
            const context = currentPrediction ? {
                prediction: currentPrediction.prediction,
                confidence: currentPrediction.confidence,
                severity: currentPrediction.severity,
                description: currentPrediction.description,
                causes: currentPrediction.causes
            } : {};

            const res = await fetch('/voice/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: query, context: context })
            });
            const data = await res.json();

            if (data.success) {
                lastVoiceResponse = data.response;
                // Use rich markdown rendering for voice response
                voiceResponse.innerHTML = renderMarkdown(data.response);
                voiceStatus.textContent = 'Response ready — tap mic to ask another question';
                speakResponseBtn.style.display = 'inline-flex';
                // Auto-speak the response (plain text version)
                if ('speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                    // Strip markdown for TTS
                    const plainText = data.response.replace(/\*\*|__|\*|_|##?\s/g, '').replace(/\n/g, ' ').substring(0, 500);
                    const utter = new SpeechSynthesisUtterance(plainText);
                    utter.rate = 0.95; utter.pitch = 1.0;
                    const voices = window.speechSynthesis.getVoices();
                    const pref = voices.find(v => v.lang.startsWith('en'));
                    if (pref) utter.voice = pref;
                    window.speechSynthesis.speak(utter);
                }
            } else {
                voiceResponse.textContent = 'Sorry, I couldn\'t process that. Try again.';
                voiceStatus.textContent = 'Error processing. Tap mic to try again.';
            }
        } catch (err) {
            voiceResponse.textContent = 'Network error. Please check your connection.';
            voiceStatus.textContent = 'Network error.';
            console.error('Voice query error:', err);
        }
    }

    // Text-to-Speech
    speakResponseBtn.addEventListener('click', () => {
        if (!lastVoiceResponse) return;

        if ('speechSynthesis' in window) {
            // Stop any ongoing speech
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(lastVoiceResponse.replace(/<[^>]*>/g, ''));
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            // Try to use a good English voice
            const voices = window.speechSynthesis.getVoices();
            const preferred = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
            if (preferred) utterance.voice = preferred;

            utterance.onstart = () => {
                speakResponseBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> Stop';
            };
            utterance.onend = () => {
                speakResponseBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Read Aloud';
            };

            window.speechSynthesis.speak(utterance);
        } else {
            showToast('Text-to-speech not supported in this browser', 'warning');
        }
    });

    // Prompt chips — click to simulate voice query
    promptChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.dataset.prompt;
            voiceTranscript.textContent = prompt;
            voiceTranscript.classList.add('active');
            processVoiceQuery(prompt);
        });
    });

    // ==================== INIT ====================
    setupAccordion();
});

/**
 * Forest Fire Early Warning System - Frontend Application
 * =========================================================
 * Vertical Scroll Layout Implementation
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let map = null;
let currentRegion = null;
let gridLayers = [];

// DOM Elements
const regionSelect = document.getElementById('region-select');
const predictBtn = document.getElementById('predict-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const toastContainer = document.getElementById('toast-container');

// Sections
const sectionOverview = document.getElementById('section-overview');
const sectionWeather = document.getElementById('section-weather');
const sectionAction = document.getElementById('section-action');
const heroStats = document.getElementById('hero-stats');

// Data Elements
const statusText = document.querySelector('.status-text');
const riskText = document.querySelector('.risk-text');
const summaryCards = document.getElementById('summary-cards');
const weatherGrid = document.getElementById('weather-grid');
const riskDriversList = document.getElementById('risk-drivers-list');
const indicesGrid = document.getElementById('indices-grid');
const actionCard = document.getElementById('recommended-action');
const headlineEl = document.getElementById('insight-headline');

// Risk Colors
const RISK_COLORS = {
    Low: '#22c55e',
    Medium: '#eab308',
    High: '#ef4444'
};

// Emergency Contacts Database
const EMERGENCY_CONTACTS = {
    'california': { country: 'USA', number: '911', local: 'CAL FIRE', region: 'California' },
    'amazon': { country: 'Brazil', number: '193', local: 'Corpo de Bombeiros', region: 'Amazonas' },
    'australia': { country: 'Australia', number: '000', local: 'RFS / CFA', region: 'New South Wales' },
    'mediterranean': { country: 'Europe', number: '112', local: 'Civil Protection', region: 'Southern Europe' },
    'default': { country: 'Global', number: '112', local: 'Emergency Services', region: 'Unknown' }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadRegions();
    setupEventListeners();
});

/**
 * Initialize Leaflet Map
 */
function initMap() {
    map = L.map('map', {
        center: [20, 0],
        zoom: 2,
        zoomControl: true,
        attributionControl: false,
        scrollWheelZoom: false
    });

    // 1. Base Layer: Satellite Imagery (Real World Texture)
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19,
        className: 'map-satellite-layer' // Used for CSS emphasis
    });

    // 2. Overlay: Borders & Labels (High Contrast)
    const labels = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CartoDB',
        subdomains: 'abcd',
        maxZoom: 19,
        zIndex: 1000 // Ensure text is always on top
    });

    // Add layers
    satellite.addTo(map);
    labels.addTo(map);
}

/**
 * Load available regions form API
 */
async function loadRegions() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/regions`);
        if (!response.ok) throw new Error('Failed to load regions');
        const data = await response.json();

        data.regions.forEach(region => {
            const option = document.createElement('option');
            option.value = region.id;
            option.textContent = region.name;
            regionSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Region load error:', error);
        addFallbackRegions();
    }
}

function addFallbackRegions() {
    const fallbackRegions = [
        { id: 'amazon', name: 'Amazon Rainforest' },
        { id: 'california', name: 'California Forests' },
        { id: 'australia', name: 'Australian Bushland' },
        { id: 'mediterranean', name: 'Mediterranean Forests' }
    ];
    fallbackRegions.forEach(region => {
        const option = document.createElement('option');
        option.value = region.id;
        option.textContent = region.name;
        regionSelect.appendChild(option);
    });
}

function setupEventListeners() {
    regionSelect.addEventListener('change', () => {
        const selected = regionSelect.value;
        predictBtn.disabled = !selected;
        if (selected) zoomToRegion(selected);
    });

    predictBtn.addEventListener('click', onPredictClick);
}

function zoomToRegion(regionId) {
    const regionBounds = {
        amazon: [[-5, -65], [-2, -60]],
        california: [[36, -121], [39, -118]],
        australia: [[-35, 149], [-32, 152]],
        mediterranean: [[37, -10], [40, -6]]
    };
    if (regionBounds[regionId]) map.fitBounds(regionBounds[regionId], { padding: [50, 50] });
}

/**
 * Handle Risk Prediction
 */
async function onPredictClick() {
    const region = regionSelect.value;
    if (!region) return;

    showLoading(true);
    clearGridLayers();

    // Reset Sections
    sectionOverview.style.display = 'none';
    sectionWeather.style.display = 'none';
    sectionAction.style.display = 'none';
    heroStats.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ region_id: region })
        });

        if (!response.ok) throw new Error('Prediction failed');
        const data = await response.json();

        // 1. Render Map Grids
        renderGrids(data.grids);

        // 2. Populate All Sections
        populateHeroStats(data.summary);
        populateOverview(data.summary);
        populateWeatherAndDrivers(data.grids, data.summary);
        populateActions(data.summary);
        populateEmergencySupport(data.summary, region);

        // 3. Reveal Sections
        revealSections();

        showToast(`Analysis Complete: ${data.summary.alert_level} Risk Detected`, 'success');

    } catch (error) {
        console.error(error);
        showToast('System Error: Could not fetch prediction data', 'error');
    } finally {
        showLoading(false);
    }
}

function renderGrids(grids) {
    grids.forEach(grid => {
        const bounds = [[grid.bounds.south, grid.bounds.west], [grid.bounds.north, grid.bounds.east]];
        const color = RISK_COLORS[grid.risk_category];
        const isHigh = grid.risk_category === 'High';

        const rect = L.rectangle(bounds, {
            color: color,
            fillColor: color,
            fillOpacity: grid.risk_score / 100 * 0.6 + 0.2,
            weight: isHigh ? 3 : 1,
            className: isHigh ? 'grid-pulse-high' : ''
        }).addTo(map);

        rect.bindTooltip(`Risk: ${grid.risk_score.toFixed(1)}%`, { permanent: false, direction: 'center' });
        gridLayers.push(rect);
    });
}

function clearGridLayers() {
    gridLayers.forEach(l => map.removeLayer(l));
    gridLayers = [];
}

/**
 * Data Population Functions
 */

function populateHeroStats(summary) {
    heroStats.style.display = 'flex';
    statusText.textContent = summary.alert_level;
    statusText.style.color = getAlertColor(summary.alert_level);
    riskText.textContent = `${summary.max_risk}%`;
    riskText.style.color = getAlertColor(summary.alert_level);
}

function populateOverview(summary) {
    // 1. Calculate Risk Value & Position
    // Risk score is 0-100. We map it to the bar width.
    const riskScore = summary.average_risk;
    const alertLevel = summary.alert_level;
    const isHigh = summary.high_risk_count > 0;

    // Update Headline (Contextual)
    headlineEl.innerHTML = `<h2>${isHigh ? '⚠️ ALERT:' : '✅ STATUS:'} ${getHeadlineText(summary)}</h2>`;
    headlineEl.style.borderLeftColor = isHigh ? RISK_COLORS.High : RISK_COLORS.Low;

    // Update Risk Bar Marker
    const marker = document.getElementById('risk-marker');
    const scoreDisplay = document.getElementById('risk-score-display');
    const levelDisplay = document.getElementById('risk-level-display');

    // Animate marker position
    setTimeout(() => {
        marker.style.left = `${Math.min(100, Math.max(0, riskScore))}%`;
    }, 100);

    // Update marker text
    scoreDisplay.textContent = `${riskScore.toFixed(1)}%`;
    scoreDisplay.style.color = getAlertColor(alertLevel);
    levelDisplay.textContent = alertLevel;

    // 2. Summary Cards (Secondary Metrics)
    summaryCards.innerHTML = `
        <div class="summary-card-item">
            <div class="label">Average Risk</div>
            <div class="value">${riskScore}%</div>
        </div>
        <div class="summary-card-item">
            <div class="label">High Risk Zones</div>
            <div class="value" style="color:${RISK_COLORS.High}">${summary.high_risk_count}</div>
        </div>
        <div class="summary-card-item">
            <div class="label">Total Area</div>
            <div class="value">${summary.total_grids * 100} km²</div>
        </div>
        <div class="summary-card-item">
            <div class="label">Data Source</div>
            <div class="value" style="font-size: 1rem; margin-top: 15px;">
                ${summary.data_source && summary.data_source.real > 0 ? '📡 Live API' : '🤖 Simulated'}
            </div>
        </div>
    `;

    // Update timestamp
    document.getElementById('last-updated').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

function populateWeatherAndDrivers(grids, summary) {
    const avg = calculateAverageWeather(grids);

    // Derived Indicators
    const fuelStatus = getFuelStatus(avg.ffmc);
    const droughtContext = getDroughtContext(avg.dc);

    // 1. Weather Grid (Two Rows)
    weatherGrid.innerHTML = `
        <div class="weather-row-label">Meteorological Conditions</div>
        <div class="weather-row compact">
            <div class="weather-card-compact">
                <span class="icon">🌡️</span>
                <span class="value">${avg.temperature.toFixed(1)}°C</span>
                <span class="label">Temp</span>
            </div>
            <div class="weather-card-compact">
                <span class="icon">💧</span>
                <span class="value">${avg.humidity.toFixed(1)}%</span>
                <span class="label">Humidity</span>
            </div>
            <div class="weather-card-compact">
                <span class="icon">💨</span>
                <span class="value">${avg.wind_speed.toFixed(1)} km/h</span>
                <span class="label">Wind</span>
            </div>
            <div class="weather-card-compact">
                <span class="icon">🌧️</span>
                <span class="value">${avg.rainfall.toFixed(1)} mm</span>
                <span class="label">Rain</span>
            </div>
        </div>

        <div class="weather-row-label">Environmental State</div>
        <div class="weather-row indicators">
            <div class="indicator-card-wide ${fuelStatus.class}">
                <div class="ind-icon">🍂</div>
                <div class="ind-content">
                    <span class="ind-value">${fuelStatus.label}</span>
                    <span class="ind-label">Fuel Moisture (FFMC)</span>
                </div>
            </div>
            <div class="indicator-card-wide ${droughtContext.class}">
                <div class="ind-icon">🏜️</div>
                <div class="ind-content">
                    <span class="ind-value">${droughtContext.label}</span>
                    <span class="ind-label">Drought Index (DC)</span>
                </div>
            </div>
        </div>
    `;

    // 2. Scientific Explainability
    const riskLevel = summary.high_risk_count > 2 ? 'HIGH' : (summary.medium_risk_count > 3 ? 'MODERATE' : 'LOW');
    document.getElementById('risk-driver-level').textContent = riskLevel;

    const drivers = generateScientificDrivers(avg);
    riskDriversList.innerHTML = drivers.map(d => `
        <li class="driver-item science-item ${d.type}">
            <span class="driver-bullet">●</span>
            <span class="driver-text">${d.text}</span>
        </li>
    `).join('');

    // 3. Overall Assessment Footer
    const assessmentBox = document.createElement('div');
    assessmentBox.className = 'assessment-footer';
    assessmentBox.innerHTML = `<strong>Overall Assessment:</strong> ${getAssessmentText(riskLevel, avg)}`;

    // Clear previous assessment if exists
    const existingFooter = document.querySelector('.drivers-box');
    const existingAssessment = existingFooter ? existingFooter.querySelector('.assessment-footer') : null;
    if (existingAssessment) existingAssessment.remove();

    if (existingFooter) { // Ensure drivers-box exists before appending
        existingFooter.appendChild(assessmentBox);
    }

    // Technical Indices (Hidden or Secondary)
    indicesGrid.innerHTML = `
        <div class="index-card"><div class="val">${avg.ffmc.toFixed(1)}</div><div class="lbl">FFMC</div></div>
        <div class="index-card"><div class="val">${avg.dmc.toFixed(1)}</div><div class="lbl">DMC</div></div>
        <div class="index-card"><div class="val">${avg.dc.toFixed(1)}</div><div class="lbl">DC</div></div>
        <div class="index-card"><div class="val">${avg.isi.toFixed(1)}</div><div class="lbl">ISI</div></div>
        <div class="index-card"><div class="val">${avg.bui.toFixed(1)}</div><div class="lbl">BUI</div></div>
        <div class="index-card"><div class="val">${avg.fwi.toFixed(1)}</div><div class="lbl">FWI</div></div>
    `;
}

// Helper: Fuel Status (Scientific)
function getFuelStatus(ffmc) {
    if (ffmc > 90) return { label: 'Critically Dry', class: 'critical' };
    if (ffmc > 85) return { label: 'Dry', class: 'warning' };
    return { label: 'Moist / Stable', class: 'safe' };
}

// Helper: Drought Context (Scientific)
function getDroughtContext(dc) {
    if (dc > 400) return { label: 'Extreme Drought', class: 'critical' };
    if (dc > 200) return { label: 'Moderate Drought', class: 'warning' };
    return { label: 'Normal Moisture', class: 'safe' };
}

// Helper: Scientific Causal Drivers
function generateScientificDrivers(avg) {
    const drivers = [];

    // Ignition Potential (Temp + Humidity)
    if (avg.temperature > 30) {
        drivers.push({ text: `High Temperature (>30°C) increases thermal energy, lowering ignition threshold.`, type: 'critical' });
    } else {
        drivers.push({ text: `Low Temperature (${avg.temperature.toFixed(1)}°C) limits evaporative potential.`, type: 'safe' });
    }

    if (avg.humidity < 30) {
        drivers.push({ text: `Critical Humidity (<30%) creates favorable atmospheric conditions for spark ignition.`, type: 'critical' });
    } else {
        drivers.push({ text: `High Relative Humidity (${avg.humidity.toFixed(1)}%) acts as a natural suppressant against ignition.`, type: 'safe' });
    }

    // Spread Potential (Wind + Fuel)
    if (avg.wind_speed > 25) {
        drivers.push({ text: `High Wind Speed (>25 km/h) indicates potential for rapid flame front expansion.`, type: 'critical' });
    } else {
        drivers.push({ text: `Low Wind Speed (<10 km/h) limits the potential for Rate of Spread (ROS).`, type: 'safe' });
    }

    // Rainfall Factor
    if (avg.rainfall > 0.5) {
        drivers.push({ text: `Recent Precipitation (${avg.rainfall.toFixed(1)}mm) has actively increased fine fuel moisture content.`, type: 'safe' });
    }

    return drivers;
}

function getAssessmentText(level, avg) {
    if (level === 'LOW') return "Current environmental conditions are not conducive to ignition or rapid fire spread. Standard monitoring protocols are sufficient.";
    if (level === 'MODERATE') return "Conditions allow for ignition in localized pockets. Situational awareness should be maintained.";
    return "Critical environmental thresholds exceeded. Probability of extreme fire behavior is high. Immediate readiness required.";
}

function populateActions(summary) {
    const isHigh = summary.high_risk_count > 0;
    const isMed = summary.medium_risk_count > 0;

    let title = "✅ Monitor Only";
    let body = "Conditions are stable. Continue standard satellite monitoring schedule. No ground action required.";
    let color = RISK_COLORS.Low;

    if (isHigh) {
        title = "🚨 DISPATCH ALERT";
        body = "Immediate mobilization recommended for high-risk zones. Deploy ground units to verified hot spots. Cancel all burn permits.";
        color = RISK_COLORS.High;
    } else if (isMed) {
        title = "⚠️ Heightened Patrol";
        body = "Increase patrol frequency in yellow zones. Standby for rapid response. Public warnings advised.";
        color = RISK_COLORS.Medium;
    }

    actionCard.innerHTML = `
        <div class="action-title" style="color:${color}">${title}</div>
        <p style="font-size: 1.1rem; line-height: 1.6;">${body}</p>
    `;
    actionCard.style.borderColor = color;
    actionCard.style.background = `linear-gradient(135deg, ${color}10, ${color}05)`;
}

function revealSections() {
    sectionOverview.style.display = 'block';
    sectionWeather.style.display = 'block';
    sectionAction.style.display = 'block';

    // Smooth scroll hint
    window.scrollTo({ top: window.innerHeight * 0.2, behavior: 'smooth' });
}

// Helpers
function showLoading(show) { loadingOverlay.style.display = show ? 'flex' : 'none'; }
function getAlertColor(level) {
    if (level === 'CRITICAL') return RISK_COLORS.High;
    if (level === 'WARNING') return RISK_COLORS.Medium;
    return RISK_COLORS.Low;
}
function getHeadlineText(summary) {
    if (summary.high_risk_count > 0) return "High fire risk detected due to critical weather conditions.";
    if (summary.medium_risk_count > 3) return "Moderate risk levels. Conditions favorable for ignition.";
    return "Fire risk is LOW today. Environment is stable.";
}
function calculateAverageWeather(grids) {
    // Simple average helper
    const sums = { temperature: 0, humidity: 0, wind_speed: 0, rainfall: 0, ffmc: 0, dmc: 0, dc: 0, isi: 0, bui: 0, fwi: 0 };
    grids.forEach(g => {
        Object.keys(sums).forEach(k => sums[k] += (g.weather[k] || 0));
    });
    const count = grids.length;
    const res = {};
    Object.keys(sums).forEach(k => res[k] = sums[k] / count);
    return res;
}

function showToast(msg, type) {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    toastContainer.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function populateEmergencySupport(summary, regionId) {
    const section = document.getElementById('section-emergency');
    const isRisk = summary.high_risk_count > 0 || summary.medium_risk_count > 0;

    // Only show if Moderate or High risk
    if (!isRisk) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';

    // Get Contact Info
    const contact = EMERGENCY_CONTACTS[regionId] || EMERGENCY_CONTACTS['default'];

    // Update DOM
    document.getElementById('emergency-location-label').textContent = `Region: ${contact.country}`;
    document.getElementById('emergency-number').textContent = contact.number;

    const alertBar = document.getElementById('emergency-alert-bar');
    const isCritical = summary.high_risk_count > 0;

    if (isCritical) {
        alertBar.className = 'emergency-alert-bar critical';
        alertBar.textContent = `🚨 CRITICAL ALERT: Immediate Fire Danger in ${contact.region}`;
    } else {
        alertBar.className = 'emergency-alert-bar';
        alertBar.textContent = `⚠️ ADVISORY: Heightened Fire Risk in ${contact.region}`;
    }

    // Setup Action Buttons
    setupEmergencyButtons(contact.number);
}

function setupEmergencyButtons(number) {
    const copyBtn = document.getElementById('btn-copy-emergency');
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(number);
        showToast('Emergency Number Copied', 'success');
    };

    const locationBtn = document.getElementById('btn-share-location');
    locationBtn.onclick = () => {
        if (navigator.share) {
            navigator.share({
                title: 'Fire Risk Emergency',
                text: `I am at risk of fire. Please send help to my coordinates.`,
                url: window.location.href
            }).catch(console.error);
        } else {
            showToast('Location sharing initiated (Simulation)', 'success');
        }
    };
}

// Top 10% yellow shade plugin for Chart.js
const top10PercentPlugin = {
    id: 'top10Percent',
    beforeDraw: (chart) => {
        const yScale = chart.scales.y;
        if (!yScale || yScale.max !== 1) return; // Only apply to 0-1 scale charts

        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        const top10Start = yScale.getPixelForValue(0.9);
        const top10End = yScale.getPixelForValue(1.0);

        ctx.save();
        ctx.fillStyle = 'rgba(251, 191, 36, 0.1)'; // Yellow with 10% opacity
        ctx.fillRect(
            chartArea.left,
            top10End,
            chartArea.right - chartArea.left,
            top10Start - top10End
        );
        ctx.restore();
    }
};
Chart.register(top10PercentPlugin);

// Vertical crosshair plugin for Chart.js
const crosshairPlugin = {
    id: 'crosshair',
    afterDraw: (chart) => {
        if (chart.tooltip?._active && chart.tooltip._active.length) {
            const activePoint = chart.tooltip._active[0];
            const ctx = chart.ctx;
            const x = activePoint.element.x;
            const topY = chart.scales.y.top;
            const bottomY = chart.scales.y.bottom;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, topY);
            ctx.lineTo(x, bottomY);
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(184, 160, 128, 0.5)';
            ctx.setLineDash([5, 3]);
            ctx.stroke();
            ctx.restore();
        }
    }
};
Chart.register(crosshairPlugin);

// Custom HTML legend plugin
const htmlLegendPlugin = {
    id: 'htmlLegend',
    afterUpdate(chart, args, options) {
        const container = document.getElementById(options.containerID || 'chart-legend');
        if (!container) return;

        container.innerHTML = '';
        const datasets = chart.data.datasets;

        datasets.forEach((dataset, index) => {
            const pill = document.createElement('span');
            pill.className = 'legend-pill';

            const color = dataset.borderColor || dataset.backgroundColor;
            const isDashed = dataset.borderDash && dataset.borderDash.length > 0;

            pill.style.borderColor = color;
            if (isDashed) {
                pill.classList.add('dashed');
            }

            pill.textContent = dataset.label;

            if (!chart.isDatasetVisible(index)) {
                pill.classList.add('strikethrough');
            }

            pill.addEventListener('click', () => {
                chart.setDatasetVisibility(index, !chart.isDatasetVisible(index));
                chart.update();
            });

            container.appendChild(pill);
        });
    }
};
Chart.register(htmlLegendPlugin);

let benchmarkData = null;
let currentPagedData = null; // Currently displayed paged data for tooltip access
let totalRuns = 0;
let totalSamples = 0;
let chart = null;
let speedChart = null;
let currentSort = { column: 'csv', direction: 'desc' };
let currentView = 'trend';
let currentChartTab = 'history';
let currentEntityTab = 'csv';
let progressInterval = null;
const ctx = document.getElementById('canvas').getContext('2d');

// Pagination state
let currentPage = 0;
const PAGE_SIZE = 50;
let savedDatasetVisibility = {}; // Track which datasets are visible when navigating
let sampleFilter = '1000plus'; // Current sample filter: 'all', 'under1000', '1000plus' (default: 1000+)
let formatFilter = 'both'; // Current format filter: 'all', 'pdf', 'csv', 'both'
let chartTabsOriginalHTML = null; // Saved original chart-tabs content for trend view

// View order for navigation
const viewOrder = ['trend', 'latest', 'entities', 'csv_entities', 'pdf_entities'];

// Format entity names: IP_ADDRESS -> IP Address, QR_CODE -> QR Code
const ACRONYMS = new Set(['IP', 'QR', 'SSN', 'URL', 'ID', 'NRP', 'AWS', 'API', 'PIN', 'VIN', 'MAC', 'UUID', 'CVV']);
function formatEntityName(name) {
    return name.split('_').map(word => {
        const upper = word.toUpperCase();
        return ACRONYMS.has(upper) ? upper : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    }).join(' ');
}

// Delta helpers
function getDelta(current, previous, isPercent = true) {
    if (previous === undefined || previous === null || current === undefined || current === null) return '';
    const diff = current - previous;
    if (Math.abs(diff) < 0.001) return '<span class="delta neutral">—</span>';
    const arrow = diff > 0 ? '▲' : '▼';
    const cls = diff > 0 ? 'up' : 'down';
    const val = isPercent ? Math.abs(diff * 100).toFixed(1) + '%' : Math.abs(diff).toFixed(1);
    return `<span class="delta ${cls}">${arrow} ${val}</span>`;
}

function getDurationDelta(current, previous) {
    if (previous === undefined || previous === null || current === undefined || current === null) return '';
    const diff = current - previous;
    if (Math.abs(diff) < 0.5) return '<span class="delta neutral">—</span>';
    const isFaster = diff < 0;
    const arrow = isFaster ? '▼' : '▲';
    const cls = isFaster ? 'up' : 'down';
    const pct = Math.abs(diff / previous * 100).toFixed(1);
    return `<span class="delta ${cls}">${arrow} ${pct}%</span>`;
}

function formatDuration(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + 's';
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${mins}m ${secs}s`;
}

function formatRunDateTime(timestamp) {
    const d = new Date(timestamp);
    if (isNaN(d)) return '';
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const h = d.getHours();
    const m = String(d.getMinutes()).padStart(2, '0');
    return `${h}:${m} ${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

function getPrevRun() {
    if (!benchmarkData || benchmarkData.length < 2) return null;
    const latest = benchmarkData[benchmarkData.length - 1];
    const latestFormat = getRunFormat(latest);
    // Find the most recent previous run with the same format
    for (let i = benchmarkData.length - 2; i >= 0; i--) {
        if (getRunFormat(benchmarkData[i]) === latestFormat) return benchmarkData[i];
    }
    return null;
}

// Pagination
// Determine the format type of a run: 'pdf', 'csv', or 'both'
function getRunFormat(run) {
    const hasCsv = (run.csv_overall_f1 != null && run.csv_overall_f1 > 0) ||
                   (run.csv_overall_recall != null && run.csv_overall_recall > 0) ||
                   (run.csv_overall_precision != null && run.csv_overall_precision > 0);
    const hasPdf = (run.pdf_overall_f1 != null && run.pdf_overall_f1 > 0) ||
                   (run.pdf_overall_recall != null && run.pdf_overall_recall > 0) ||
                   (run.pdf_overall_precision != null && run.pdf_overall_precision > 0);
    if (hasCsv && hasPdf) return 'both';
    if (hasPdf) return 'pdf';
    return 'csv';
}

function getFilteredData() {
    if (!benchmarkData || benchmarkData.length === 0) return [];

    return benchmarkData.filter(run => {
        // Sample size filter
        if (sampleFilter === 'under1000' && run.samples >= 1000) return false;
        if (sampleFilter === '1000plus' && run.samples < 1000) return false;

        // Format filter
        if (formatFilter !== 'all') {
            if (getRunFormat(run) !== formatFilter) return false;
        }

        return true;
    });
}

function filterByFormat(format) {
    formatFilter = format;
    document.querySelectorAll('.format-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.format === format);
    });
    currentPage = 0;
    toggleView(currentView);
}

function getPagedData() {
    const filteredData = getFilteredData();
    if (filteredData.length === 0) return [];
    const start = Math.max(0, filteredData.length - PAGE_SIZE - (currentPage * PAGE_SIZE));
    const end = filteredData.length - (currentPage * PAGE_SIZE);
    return filteredData.slice(start, end);
}

function getTotalPages() {
    const filteredData = getFilteredData();
    return Math.ceil(filteredData.length / PAGE_SIZE);
}

function renderPaginationControls() {
    const container = document.getElementById('pagination-controls');
    const totalPages = getTotalPages();
    const filteredData = getFilteredData();
    if ((currentView !== 'trend' && currentView !== 'speed') || totalPages <= 1) {
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');
    const startRun = filteredData.length - PAGE_SIZE - (currentPage * PAGE_SIZE) + 1;
    const endRun = Math.min(filteredData.length - (currentPage * PAGE_SIZE), filteredData.length);
    container.innerHTML = `
        <button class="pagination-btn" onclick="goToPage(${currentPage + 1})" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>← Older</button>
        <span class="pagination-info">Runs ${Math.max(1, startRun).toLocaleString()}-${endRun.toLocaleString()} of ${filteredData.length.toLocaleString()}</span>
        <button class="pagination-btn" onclick="goToPage(${currentPage - 1})" ${currentPage <= 0 ? 'disabled' : ''}>Newer →</button>
    `;
}

function hidePagination() {
    document.getElementById('pagination-controls').classList.add('hidden');
}

function goToPage(page) {
    const totalPages = getTotalPages();
    if (page < 0 || page >= totalPages) return;

    // Save current dataset visibility state before navigating
    if (chart && chart.data && chart.data.datasets) {
        savedDatasetVisibility = {};
        chart.data.datasets.forEach((dataset, index) => {
            savedDatasetVisibility[dataset.label] = chart.isDatasetVisible(index);
        });
    }

    currentPage = page;
    toggleView(currentView);
}

function filterBySamples() {
    const select = document.getElementById('sample-filter');
    sampleFilter = select.value;
    currentPage = 0; // Reset to first page when filter changes
    toggleView(currentView); // Refresh the current view
}

// Progress banner collapsed state (closed by default)
let progressBannerCollapsed = true;

function toggleProgressBanner() {
    progressBannerCollapsed = !progressBannerCollapsed;
    const banner = document.getElementById('progress-banner');
    if (progressBannerCollapsed) {
        banner.classList.remove('visible');
    } else {
        banner.classList.add('visible');
    }
}

function updateJobStatusButton(isRunning, phaseText, progress) {
    const jobBtn = document.getElementById('job-status-btn');
    const jobText = document.getElementById('job-status-text');
    const jobPercent = document.getElementById('job-status-percent');

    if (isRunning) {
        jobBtn.classList.remove('hidden');
        // Shorten the phase text for the button
        let shortText = phaseText || 'Running...';
        if (shortText.length > 35) {
            shortText = shortText.substring(0, 33) + '...';
        }
        jobText.textContent = shortText;
        const pct = progress || 0;
        jobPercent.textContent = pct + '%';
        jobPercent.style.setProperty('--progress', pct + '%');
    } else {
        jobBtn.classList.add('hidden');
        progressBannerCollapsed = true;
    }
}

// Progress UI
function updateProgressUI(data) {
    const banner = document.getElementById('progress-banner');
    if (!progressBannerCollapsed) {
        banner.classList.add('visible');
    }
    const isComplete = data.status === 'complete' || data.status === 'stopped';
    const isRunning = data.status === 'running' || data.status === 'starting';
    banner.classList.toggle('progress-complete', isComplete);

    const stopBtn = document.getElementById('stop-test-btn');
    const runBtn = document.getElementById('run-test-btn');
    if (isRunning) {
        stopBtn.classList.remove('hidden');
        runBtn.classList.add('hidden');
        runBtn.disabled = true;
    } else {
        stopBtn.classList.add('hidden');
        runBtn.classList.remove('hidden');
        runBtn.disabled = false;
        runBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Test`;
    }

    const phaseText = data.status === 'stopped' ? 'Benchmark Stopped' :
        isComplete ? 'Benchmark Complete!' : (data.phase || 'Running...');
    document.getElementById('progress-phase').textContent = phaseText;

    const progress = data.progress || 0;

    // Update job status button
    updateJobStatusButton(isRunning, phaseText, progress);
    document.getElementById('progress-bar').style.width = progress + '%';
    document.getElementById('progress-text').textContent = progress + '%';

    let elapsed = data.elapsed_seconds || 0;
    if (data.start_time && !isComplete) {
        elapsed = (new Date() - new Date(data.start_time)) / 1000;
    }
    const mins = Math.floor(elapsed / 60);
    const secs = Math.floor(elapsed % 60);
    const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

    const samplesProcessed = data.samples_processed || 0;
    const totalSamples = data.total_samples || 0;
    const totalSets = data.total_sets || Math.ceil(totalSamples / 100) || 0;
    const currentSet = data.current_set || 0;
    const detections = data.detections || 0;
    const speed = data.samples_per_second || 0;
    const currentSampleInSet = data.current_sample_in_set || 0;
    const samplesInCurrentSet = data.samples_in_current_set || 100;

    document.getElementById('progress-samples').textContent = `${samplesProcessed.toLocaleString()} / ${totalSamples.toLocaleString()}`;
    document.getElementById('progress-sets').textContent = `${currentSet} / ${totalSets}`;

    const setProgressElem = document.getElementById('progress-set-samples');
    if (setProgressElem) {
        if (currentSampleInSet > 0 && currentSet > 0) setProgressElem.textContent = `${currentSampleInSet}/${samplesInCurrentSet}`;
        else if (currentSet > 0) setProgressElem.textContent = `Set ${currentSet}`;
        else setProgressElem.textContent = '--';
    }

    document.getElementById('progress-detections').textContent = detections.toLocaleString();
    document.getElementById('progress-speed').textContent = speed > 0 ? speed.toFixed(2) : '0.00';
    document.getElementById('progress-elapsed').textContent = timeStr;

    if (isComplete) {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }

        setTimeout(async () => {
            try {
                const historyResponse = await fetch('benchmark_history/benchmark_history.json?t=' + Date.now());
                if (historyResponse.ok) loadHistoryData(await historyResponse.json());
            } catch (err) {}
        }, 1000);
    }
}

function loadHistoryData(json) {
    if (!json.runs || !Array.isArray(json.runs) || json.runs.length === 0) return;
    benchmarkData = json.runs;
    totalRuns = json.total_runs || json.runs.length;
    totalSamples = json.total_samples || json.runs.reduce((sum, r) => sum + r.samples, 0);
    renderSummaryStats();
    toggleView('trend');
}

// Auto-load
async function autoLoadFiles() {
    try {
        const historyResponse = await fetch('benchmark_history/benchmark_history.json?t=' + Date.now());
        if (historyResponse.ok) loadHistoryData(await historyResponse.json());
    } catch (err) {}
    startProgressMonitoring();
}

async function startProgressMonitoring() {
    let lastReportedElapsed = null;
    let unchangedCount = 0;

    try {
        const progressResponse = await fetch('benchmark_history/benchmark_progress.json?t=' + Date.now());
        if (progressResponse.ok) {
            const progressJson = await progressResponse.json();
            const startTime = progressJson.start_time ? new Date(progressJson.start_time) : null;
            const elapsedSinceStart = startTime ? (new Date() - startTime) / 1000 : 0;
            const isOldStart = elapsedSinceStart > 1800;
            const isStale = isOldStart && progressJson.status !== 'running' && progressJson.status !== 'starting';
            const isRecentlyComplete = (progressJson.status === 'complete' || progressJson.status === 'stopped') && elapsedSinceStart < 300;
            const isActive = (progressJson.status === 'running' || progressJson.status === 'starting');

            if ((isActive && !isStale) || isRecentlyComplete) {
                updateProgressUI(progressJson);
                lastReportedElapsed = progressJson.elapsed_seconds;

                if (progressJson.status !== 'complete') {
                    if (progressInterval) clearInterval(progressInterval);
                    progressInterval = setInterval(async () => {
                        try {
                            const resp = await fetch('benchmark_history/benchmark_progress.json?t=' + Date.now());
                            if (resp.ok) {
                                const data = await resp.json();
                                if (data.elapsed_seconds === lastReportedElapsed) unchangedCount++;
                                else { unchangedCount = 0; lastReportedElapsed = data.elapsed_seconds; }

                                if (unchangedCount >= 120 && data.status !== 'running' && data.status !== 'starting') {
                                    document.getElementById('progress-banner').classList.remove('visible');
                                    document.getElementById('job-status-btn').classList.add('hidden');
                                    clearInterval(progressInterval);
                                } else updateProgressUI(data);
                            }
                        } catch (e) {}
                    }, 500);
                }
            } else {
                document.getElementById('progress-banner').classList.remove('visible');
                document.getElementById('job-status-btn').classList.add('hidden');
            }
        }
    } catch (err) {}
}

// Background monitoring
let backgroundMonitorInterval = null;
let lastKnownStatus = null;
let lastKnownStartTime = null;

function startBackgroundMonitoring() {
    if (backgroundMonitorInterval) return;
    backgroundMonitorInterval = setInterval(async () => {
        try {
            const resp = await fetch('benchmark_history/benchmark_progress.json?t=' + Date.now());
            if (!resp.ok) return;
            const data = await resp.json();
            const isRunning = data.status === 'running' || data.status === 'starting';
            const isNewRun = data.start_time !== lastKnownStartTime && isRunning;
            const statusChangedToRunning = !lastKnownStatus?.match(/running|starting/) && isRunning;

            if (isNewRun || statusChangedToRunning) {
                updateProgressUI(data);
                startProgressMonitoring();
            }
            lastKnownStatus = data.status;
            lastKnownStartTime = data.start_time;
        } catch (e) {}
    }, 2000);
}

// Drag scroll
function initDragScroll() {
    document.querySelectorAll('.stats-grid').forEach(container => {
        let isDown = false, startX, scrollLeft;

        container.addEventListener('mousedown', (e) => {
            if (e.target.closest('.stat-card')) return;
            isDown = true;
            container.style.cursor = 'grabbing';
            startX = e.pageX - container.offsetLeft;
            scrollLeft = container.scrollLeft;
        });
        container.addEventListener('mouseleave', () => { isDown = false; container.style.cursor = 'grab'; });
        container.addEventListener('mouseup', () => { isDown = false; container.style.cursor = 'grab'; });
        container.addEventListener('mousemove', (e) => {
            if (!isDown) return;
            e.preventDefault();
            container.scrollLeft = scrollLeft - ((e.pageX - container.offsetLeft - startX) * 1.5);
        });
        container.addEventListener('scroll', updateChevronVisibility);
    });
    setTimeout(updateChevronVisibility, 100);
}

function scrollStats(direction) {
    const currentIndex = viewOrder.indexOf(currentView);
    let newIndex = currentIndex + direction;
    if (newIndex >= viewOrder.length) newIndex = 0;
    if (newIndex < 0) newIndex = viewOrder.length - 1;
    toggleView(viewOrder[newIndex]);
    setTimeout(() => {
        const activeCard = document.querySelector('.stat-card.active');
        if (activeCard) activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }, 50);
}

function updateChevronVisibility() {
    const leftBtn = document.getElementById('scroll-left');
    const rightBtn = document.getElementById('scroll-right');
    if (leftBtn) leftBtn.classList.remove('hidden');
    if (rightBtn) rightBtn.classList.remove('hidden');
}

// Drawer
function openRunDrawer() {
    document.getElementById('drawer-overlay').classList.add('visible');
    document.getElementById('run-drawer').classList.add('open');
}

function closeRunDrawer() {
    document.getElementById('drawer-overlay').classList.remove('visible');
    document.getElementById('run-drawer').classList.remove('open');
}

function toggleLoopCount() {
    document.getElementById('loop-count-container').style.display = document.getElementById('loop-mode').checked ? 'flex' : 'none';
}

function toggleLLMModels() {
    const mode = document.querySelector('input[name="bench-mode"]:checked').value;
    const showLLM = mode === 'llm' || mode === 'both';
    document.getElementById('llm-models-group').style.display = showLLM ? 'block' : 'none';
    document.getElementById('llm-prompt-group').style.display = showLLM ? 'block' : 'none';
}

async function startBenchmark() {
    const samples = document.getElementById('sample-count').value;
    const format = document.querySelector('input[name="test-format"]:checked').value;
    const benchMode = document.querySelector('input[name="bench-mode"]:checked').value;
    const fastMode = document.getElementById('fast-mode').checked;
    const keepFiles = document.getElementById('keep-files').checked;
    const saveFeedback = document.getElementById('save-feedback').checked;
    const datasetCheckboxes = document.querySelectorAll('.form-group .checkbox-item input[id^="dataset-"]:checked');
    const selectedDatasets = Array.from(datasetCheckboxes).map(cb => cb.value);

    if (selectedDatasets.length === 0) { alert('Please select at least one dataset'); return; }

    const loopMode = document.getElementById('loop-mode').checked;
    const loopCount = document.getElementById('loop-count').value;

    // LLM comparison mode
    if (benchMode === 'llm' || benchMode === 'both') {
        const llmCheckboxes = document.querySelectorAll('#llm-models-group .checkbox-item input:checked');
        const selectedModels = Array.from(llmCheckboxes).map(cb => cb.value);
        const promptStyle = document.querySelector('input[name="prompt-style"]:checked').value;

        if (selectedModels.length === 0 && benchMode === 'llm') { alert('Please select at least one LLM model'); return; }

        let llmArgs = `--datasets ${selectedDatasets.join(',')} --samples ${samples}`;
        if (selectedModels.length > 0) llmArgs += ` --models ${selectedModels.join(',')}`;
        if (benchMode === 'llm') llmArgs += ' --hush-only';  // Skip hush if llm-only... actually opposite
        if (promptStyle === 'few-shot') llmArgs += ' --few-shot';

        // For "both" mode, we need hush + LLMs; for "llm" mode, just LLMs (hush always runs as baseline)
        const llmCmd = `python3 tests/benchmark_llm_comparison.py ${llmArgs}`;

        closeRunDrawer();
        const runBtn = document.getElementById('run-test-btn');
        runBtn.classList.add('hidden');
        runBtn.disabled = true;
        const banner = document.getElementById('progress-banner');
        banner.classList.add('visible');
        updateProgressUI({ status: 'running', phase: 'LLM Comparison Benchmark', progress: 5, total_samples: parseInt(samples), samples_processed: 0, detections: 0, start_time: new Date().toISOString(), elapsed_seconds: 0 });
        document.getElementById('progress-phase').innerHTML = `<span style="font-size: 0.8rem; color: var(--muted);">Run in terminal:</span><br><code style="background: rgba(0,0,0,0.3); padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.8rem; display: inline-block; margin-top: 0.5rem; cursor: pointer;" onclick="navigator.clipboard.writeText('${llmCmd}'); this.style.background='rgba(74,222,128,0.2)';" title="Click to copy">${llmCmd}</code>`;

        // Also try to start via API
        try {
            await fetch('/api/benchmark/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ samples: parseInt(samples), datasets: selectedDatasets, models: selectedModels, prompt_style: promptStyle, bench_mode: benchMode, args: llmArgs, llm_benchmark: true })
            });
        } catch (e) { /* fallback to terminal command shown above */ }
        startProgressMonitoring();
        return;
    }

    // Standard Hush Engine benchmark
    let args = `--datasets ${selectedDatasets.join(',')} --samples ${samples}`;
    if (format === 'csv') args += ' --no-pdf';
    if (format === 'pdf') args += ' --pdf-only';
    if (fastMode) args += ' --fast';
    if (keepFiles) args += ' --keep-files';
    if (saveFeedback) args += ' --save-feedback';
    if (loopMode && loopCount > 1) args += ` --loops ${loopCount}`;

    closeRunDrawer();

    const runBtn = document.getElementById('run-test-btn');
    runBtn.classList.add('hidden');
    runBtn.disabled = true;

    const banner = document.getElementById('progress-banner');
    banner.classList.add('visible');
    updateProgressUI({ status: 'starting', phase: 'Initializing benchmark...', progress: 0, total_samples: parseInt(samples), samples_processed: 0, detections: 0, start_time: new Date().toISOString(), elapsed_seconds: 0 });

    try {
        const response = await fetch('/api/benchmark/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: parseInt(samples), format, datasets: selectedDatasets, fast_mode: fastMode, keep_files: keepFiles, save_feedback: saveFeedback, args })
        });
        if (!response.ok) throw new Error('API not available');
    } catch (error) {
        updateProgressUI({ status: 'running', phase: 'Run this command in terminal:', progress: 5, total_samples: parseInt(samples), samples_processed: 0, detections: 0, start_time: new Date().toISOString(), elapsed_seconds: 0 });
        document.getElementById('progress-phase').innerHTML = `<span style="font-size: 0.8rem; color: var(--muted);">Run in terminal:</span><br><code style="background: rgba(0,0,0,0.3); padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.8rem; display: inline-block; margin-top: 0.5rem; cursor: pointer;" onclick="navigator.clipboard.writeText('python3 tests/benchmark_accuracy.py ${args}'); this.style.background='rgba(74,222,128,0.2)';" title="Click to copy">python3 tests/benchmark_accuracy.py ${args}</code>`;
        startProgressMonitoring();
    }
}

async function stopBenchmark() {
    const stopBtn = document.getElementById('stop-test-btn');
    const originalText = stopBtn.innerHTML;
    stopBtn.disabled = true;
    stopBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spin"><circle cx="12" cy="12" r="10" stroke-dasharray="40" stroke-dashoffset="10"></circle></svg> Stopping...`;

    try {
        await fetch('/api/benchmark/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        stopBtn.classList.add('hidden');
        document.getElementById('progress-banner').classList.add('progress-complete');
        document.getElementById('progress-phase').textContent = 'Benchmark Stopped';
        document.getElementById('run-test-btn').classList.remove('hidden');
        document.getElementById('run-test-btn').disabled = false;
        document.getElementById('run-test-btn').innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Test`;
        if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
    } catch (error) {
        stopBtn.disabled = false;
        stopBtn.innerHTML = originalText;
    }
}

// Stats rendering
function renderSummaryStats() {
    const latest = benchmarkData[benchmarkData.length - 1];
    const prev = getPrevRun();

    const csvPrecision = latest.csv_overall_precision || 0;
    const csvRecall = latest.csv_overall_recall || 0;
    const csvF1 = latest.csv_overall_f1 || 0;
    const pdfPrecision = latest.pdf_overall_precision;
    const pdfRecall = latest.pdf_overall_recall;
    const pdfF1 = latest.pdf_overall_f1;

    const prevCsvPrecision = prev ? (prev.csv_overall_precision || 0) : null;
    const prevCsvRecall = prev ? (prev.csv_overall_recall || 0) : null;
    const prevCsvF1 = prev ? (prev.csv_overall_f1 || 0) : null;
    const prevPdfPrecision = prev ? prev.pdf_overall_precision : null;
    const prevPdfRecall = prev ? prev.pdf_overall_recall : null;
    const prevPdfF1 = prev ? prev.pdf_overall_f1 : null;

    // Detect PDF-only runs (CSV metrics are null/zero, but PDF metrics exist)
    const isPdfOnly = (latest.csv_overall_f1 == null || latest.csv_overall_f1 === 0) &&
                       (latest.csv_overall_recall == null || latest.csv_overall_recall === 0) &&
                       (pdfF1 != null || pdfRecall != null);

    const getClass = val => Math.round(val * 1000) >= 900 ? 'green' : '';
    const csvPrecisionClass = getClass(csvPrecision);
    const csvRecallClass = getClass(csvRecall);
    const csvF1Class = getClass(csvF1);
    const pdfPrecisionClass = pdfPrecision != null ? getClass(pdfPrecision) : '';
    const pdfRecallClass = pdfRecall != null ? getClass(pdfRecall) : '';
    const pdfF1Class = pdfF1 != null ? getClass(pdfF1) : '';

    const csvPrecisionDelta = prev && prevCsvPrecision ? getDelta(csvPrecision, prevCsvPrecision) : '';
    const csvRecallDelta = prev ? getDelta(csvRecall, prevCsvRecall) : '';
    const csvF1Delta = prev && prevCsvF1 ? getDelta(csvF1, prevCsvF1) : '';
    const pdfPrecisionDelta = prev && prevPdfPrecision != null && pdfPrecision != null ? getDelta(pdfPrecision, prevPdfPrecision) : '';
    const pdfRecallDelta = prev && prevPdfRecall != null && pdfRecall != null ? getDelta(pdfRecall, prevPdfRecall) : '';
    const pdfF1Delta = prev && prevPdfF1 != null && pdfF1 != null ? getDelta(pdfF1, prevPdfF1) : '';
    const durationDelta = prev && prev.duration_seconds && latest.duration_seconds ? getDurationDelta(latest.duration_seconds, prev.duration_seconds) : '';

    // Swap display for PDF-only runs
    let precisionHtml, recallHtml, f1Html;
    if (isPdfOnly) {
        precisionHtml = `
            <div class="stat-label">Precision</div>
            <div class="stat-value ${pdfPrecisionClass}">${pdfPrecision != null ? (pdfPrecision * 100).toFixed(1) + '%' : '0.0%'}${pdfPrecisionDelta}</div>
            <div class="stat-sub">CSV: <span class="${csvPrecisionClass}">${(csvPrecision * 100).toFixed(1)}%</span>${csvPrecisionDelta}</div>
        `;
        recallHtml = `
            <div class="stat-label">Recall</div>
            <div class="stat-value ${pdfRecallClass}">${pdfRecall != null ? (pdfRecall * 100).toFixed(1) + '%' : '0.0%'}${pdfRecallDelta}</div>
            <div class="stat-sub">CSV: <span class="${csvRecallClass}">${(csvRecall * 100).toFixed(1)}%</span>${csvRecallDelta}</div>
        `;
        f1Html = `
            <div class="stat-label">F1 Score</div>
            <div class="stat-value ${pdfF1Class}">${pdfF1 != null ? (pdfF1 * 100).toFixed(1) + '%' : '0.0%'}${pdfF1Delta}</div>
            <div class="stat-sub">CSV: <span class="${csvF1Class}">${(csvF1 * 100).toFixed(1)}%</span>${csvF1Delta}</div>
        `;
    } else {
        precisionHtml = `
            <div class="stat-label">Precision</div>
            <div class="stat-value ${csvPrecisionClass}">${(csvPrecision * 100).toFixed(1)}%${csvPrecisionDelta}</div>
            <div class="stat-sub">PDF: <span class="${pdfPrecisionClass}">${pdfPrecision != null ? (pdfPrecision * 100).toFixed(1) + '%' : '--'}</span>${pdfPrecisionDelta}</div>
        `;
        recallHtml = `
            <div class="stat-label">Recall</div>
            <div class="stat-value ${csvRecallClass}">${(csvRecall * 100).toFixed(1)}%${csvRecallDelta}</div>
            <div class="stat-sub">PDF: <span class="${pdfRecallClass}">${pdfRecall != null ? (pdfRecall * 100).toFixed(1) + '%' : '--'}</span>${pdfRecallDelta}</div>
        `;
        f1Html = `
            <div class="stat-label">F1 Score</div>
            <div class="stat-value ${csvF1Class}">${(csvF1 * 100).toFixed(1)}%${csvF1Delta}</div>
            <div class="stat-sub">PDF: <span class="${pdfF1Class}">${pdfF1 != null ? (pdfF1 * 100).toFixed(1) + '%' : '--'}</span>${pdfF1Delta}</div>
        `;
    }

    document.getElementById('stats-grid').innerHTML = `
        <div class="stat-card" id="card-runs" onclick="toggleView('trend')" title="Click for Metrics Trend">
            <div class="stat-label">Total Runs</div>
            <div class="stat-value">${totalRuns.toLocaleString()}</div>
            <div class="stat-sub">${totalSamples.toLocaleString()} samples</div>
        </div>
        <div class="stat-card" id="card-duration" onclick="toggleView('latest')" title="Click for Latest Stats">
            <div class="stat-label">Latest Duration</div>
            <div class="stat-value">${latest.duration_seconds ? formatDuration(latest.duration_seconds) : 'N/A'}${durationDelta}</div>
            <div class="stat-sub">${latest.samples.toLocaleString()} samples</div>
        </div>
        <div class="stat-card" id="card-precision" onclick="toggleView('entities')" title="Click for Entity Breakdown">
            ${precisionHtml}
        </div>
        <div class="stat-card" id="card-recall" onclick="toggleView('csv_entities')" title="Click for Recall by Entity (CSV vs PDF)">
            ${recallHtml}
        </div>
        <div class="stat-card" id="card-f1" onclick="toggleView('pdf_entities')" title="Click for F1 by Entity (CSV vs PDF)">
            ${f1Html}
        </div>
    `;
    updateActiveCard();
    setTimeout(updateChevronVisibility, 50);
}

function updateActiveCard() {
    document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
    const cardMap = { 'trend': 'card-runs', 'entities': 'card-precision', 'csv_entities': 'card-recall', 'pdf_entities': 'card-f1', 'latest': 'card-duration' };
    const activeCard = document.getElementById(cardMap[currentView]);
    if (activeCard) {
        activeCard.classList.add('active');
        // Scroll the card into view if it's partially cut off
        const container = document.getElementById('stats-grid');
        if (container) {
            const cardCenter = activeCard.offsetLeft + activeCard.offsetWidth / 2;
            const targetScroll = cardCenter - container.offsetWidth / 2;
            container.scrollTo({ left: targetScroll, behavior: 'smooth' });
        }
    }
}

function toggleView(viewType) {
    if (chart) chart.destroy();
    chart = null;
    currentView = viewType;
    updateActiveCard();
    updateChevronVisibility();
    hidePagination();

    const chartContainer = document.getElementById('chart-container');
    const tableContainer = document.getElementById('table-container');
    const legendContainer = document.getElementById('chart-legend');
    const speedChartSection = document.getElementById('speed-chart-section');
    const chartTabs = document.getElementById('chart-tabs');

    // Save original chart-tabs content on first call
    if (chartTabs && !chartTabsOriginalHTML) chartTabsOriginalHTML = chartTabs.innerHTML;

    // Hide speed chart and tabs by default (shown for trend view)
    if (speedChartSection) speedChartSection.classList.add('hidden');
    if (chartTabs) chartTabs.classList.add('hidden');

    if (viewType === 'latest') {
        chartContainer.classList.add('hidden');
        tableContainer.classList.remove('hidden');
        if (legendContainer) {
            legendContainer.innerHTML = '';
            legendContainer.classList.add('hidden');
        }
        renderLatestStatsGrid();
        return;
    }

    chartContainer.classList.remove('hidden');
    tableContainer.classList.add('hidden');
    if (legendContainer) legendContainer.classList.remove('hidden');

    const latest = benchmarkData[benchmarkData.length - 1];

    if (viewType === 'trend') {
        const pagedData = getPagedData();
        currentPagedData = pagedData; // Store for tooltip access
        const filteredData = getFilteredData();
        const startIndex = filteredData.length - pagedData.length - (currentPage * PAGE_SIZE) + 1;
        // Show 3px round dots only for runs with 1,000+ samples
        const highSamplePointRadius = pagedData.map(r => r.samples >= 1000 ? 3 : 0);
        // Compute EMA-smoothed normalized F1 trend for the overlay (uses both CSV and PDF)
        const f1Raw = pagedData.map(r => {
            const csv = r.csv_overall_f1 != null ? r.csv_overall_f1 : null;
            const pdf = r.pdf_overall_f1 != null ? r.pdf_overall_f1 : null;
            if (csv != null && pdf != null) return (csv + pdf) / 2;
            return csv != null ? csv : pdf;
        });
        const alpha = 0.15;
        const f1Ema = [];
        f1Raw.forEach((v, i) => {
            if (v == null) {
                f1Ema.push(null);
            } else {
                // Find the last non-null EMA value
                let prevEma = null;
                for (let j = i - 1; j >= 0; j--) {
                    if (f1Ema[j] != null) {
                        prevEma = f1Ema[j];
                        break;
                    }
                }
                f1Ema.push(prevEma == null ? v : alpha * v + (1 - alpha) * prevEma);
            }
        });
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: pagedData.map((r, i) => r.llm_benchmark ? `${startIndex + i} (${r.model_display_name || 'LLM'})` : `${startIndex + i}`),
                datasets: [
                    { label: 'F1 Trend', data: f1Ema, borderColor: 'rgba(245, 158, 11, 0.5)', backgroundColor: 'transparent', tension: 0.4, pointRadius: 0, pointHoverRadius: 0, fill: false, borderWidth: 1, order: 10 },
                    { label: 'Recall', data: pagedData.map(r => r.csv_overall_recall != null ? r.csv_overall_recall : null), borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', tension: 0.3, pointRadius: pagedData.map(r => r.llm_benchmark ? 8 : highSamplePointRadius), pointHoverRadius: 6, fill: false, borderWidth: 1, pointStyle: pagedData.map(r => r.llm_benchmark ? 'triangle' : 'circle'), pointBackgroundColor: pagedData.map(r => r.llm_benchmark ? '#e74c3c' : '#10b981'), segment: { borderDash: ctx => (pagedData[ctx.p0DataIndex]?.csv_overall_recall == null || pagedData[ctx.p1DataIndex]?.csv_overall_recall == null) ? [5, 5] : [] } },
                    { label: 'F1 Score', data: pagedData.map(r => r.csv_overall_f1 != null ? r.csv_overall_f1 : null), borderColor: '#f59e0b', backgroundColor: 'transparent', tension: 0.3, pointRadius: pagedData.map(r => r.llm_benchmark ? 8 : highSamplePointRadius), pointHoverRadius: 6, fill: false, borderWidth: 1, pointStyle: pagedData.map(r => r.llm_benchmark ? 'triangle' : 'circle'), pointBackgroundColor: pagedData.map(r => r.llm_benchmark ? '#e74c3c' : '#f59e0b'), segment: { borderDash: ctx => (pagedData[ctx.p0DataIndex]?.csv_overall_f1 == null || pagedData[ctx.p1DataIndex]?.csv_overall_f1 == null) ? [5, 5] : [] } },
                    { label: 'Precision', data: pagedData.map(r => r.csv_overall_precision != null ? r.csv_overall_precision : null), borderColor: '#6366f1', backgroundColor: 'rgba(99, 102, 241, 0.1)', tension: 0.3, pointRadius: pagedData.map(r => r.llm_benchmark ? 8 : highSamplePointRadius), pointHoverRadius: 6, fill: false, borderWidth: 1, pointStyle: pagedData.map(r => r.llm_benchmark ? 'triangle' : 'circle'), pointBackgroundColor: pagedData.map(r => r.llm_benchmark ? '#e74c3c' : '#6366f1'), segment: { borderDash: ctx => (pagedData[ctx.p0DataIndex]?.csv_overall_precision == null || pagedData[ctx.p1DataIndex]?.csv_overall_precision == null) ? [5, 5] : [] } },
                    { label: 'PDF Recall', data: pagedData.map(r => r.pdf_overall_recall || null), borderColor: '#10b981', backgroundColor: 'transparent', tension: 0.3, pointRadius: highSamplePointRadius, pointHoverRadius: 4, fill: false, borderWidth: 1, borderDash: [5, 5] },
                    { label: 'PDF F1', data: pagedData.map(r => r.pdf_overall_f1 || null), borderColor: '#f59e0b', backgroundColor: 'transparent', tension: 0.3, pointRadius: highSamplePointRadius, pointHoverRadius: 4, fill: false, borderWidth: 1, borderDash: [5, 5] },
                    { label: 'PDF Precision', data: pagedData.map(r => r.pdf_overall_precision || null), borderColor: '#6366f1', backgroundColor: 'transparent', tension: 0.3, pointRadius: highSamplePointRadius, pointHoverRadius: 4, fill: false, borderWidth: 1, borderDash: [5, 5] }
                ]
            },
            options: getChartOptions(0, 1.0, true),
            plugins: [{
                id: 'thresholdLine',
                afterDraw(chart) {
                    const yScale = chart.scales.y;
                    const y = yScale.getPixelForValue(0.9);
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.beginPath();
                    ctx.setLineDash([4, 4]);
                    ctx.strokeStyle = '#eab308';
                    ctx.lineWidth = 1;
                    ctx.moveTo(chart.chartArea.left, y);
                    ctx.lineTo(chart.chartArea.right, y);
                    ctx.stroke();
                    ctx.restore();
                }
            }]
        });

        // Restore saved dataset visibility state after navigation
        if (savedDatasetVisibility && Object.keys(savedDatasetVisibility).length > 0) {
            chart.data.datasets.forEach((dataset, index) => {
                if (savedDatasetVisibility.hasOwnProperty(dataset.label)) {
                    chart.setDatasetVisibility(index, savedDatasetVisibility[dataset.label]);
                }
            });
            chart.update('none'); // Update without animation
        }

        // Apply format filter visibility (overrides saved state for hidden groups)
        if (formatFilter === 'csv' || formatFilter === 'pdf') {
            const csvLabels = new Set(['Recall', 'F1 Score', 'Precision']);
            const pdfLabels = new Set(['PDF Recall', 'PDF F1', 'PDF Precision']);
            chart.data.datasets.forEach((dataset, index) => {
                if (dataset.label === 'F1 Trend') return; // always visible (uses both CSV + PDF)
                if (formatFilter === 'csv' && pdfLabels.has(dataset.label)) {
                    chart.setDatasetVisibility(index, false);
                } else if (formatFilter === 'pdf' && csvLabels.has(dataset.label)) {
                    chart.setDatasetVisibility(index, false);
                }
            });
            chart.update('none');
        }

        renderPaginationControls();
        // Restore original chart tabs (filters) and show both charts
        if (chartTabs) {
            if (chartTabsOriginalHTML) chartTabs.innerHTML = chartTabsOriginalHTML;
            chartTabs.classList.remove('hidden');
            // Re-apply current filter state
            chartTabs.querySelectorAll('.format-pill').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.format === formatFilter);
            });
            const sampleSelect = chartTabs.querySelector('#sample-filter');
            if (sampleSelect) sampleSelect.value = sampleFilter;
        }
    } else if (viewType === 'entities') {
        chartContainer.classList.add('hidden');
        tableContainer.classList.remove('hidden');
        if (legendContainer) {
            legendContainer.innerHTML = '';
            legendContainer.classList.add('hidden');
        }
        // Show chart-tabs bar with entity pills (consistent layout with trend view)
        if (chartTabs) {
            chartTabs.classList.remove('hidden');
            renderEntityTabs(chartTabs);
        }
        renderLatestTable();
        return;
    } else if (viewType === 'csv_entities') {
        const csvMetrics = latest.csv_metrics_by_type || latest.csv_recall_by_type || {};
        const pdfMetrics = latest.pdf_metrics_by_type || latest.pdf_recall_by_type || {};
        const allEntities = [...new Set([...Object.keys(csvMetrics), ...Object.keys(pdfMetrics)])];
        const sortedEntities = allEntities.sort((a, b) => (csvMetrics[b]?.recall || 0) - (csvMetrics[a]?.recall || 0));
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedEntities.map(formatEntityName),
                datasets: [
                    { label: 'CSV Recall', data: sortedEntities.map(e => csvMetrics[e]?.recall || 0), backgroundColor: 'rgba(99, 102, 241, 0.6)', borderColor: '#6366f1', borderWidth: 1 },
                    { label: 'PDF Recall', data: sortedEntities.map(e => pdfMetrics[e]?.recall || 0), backgroundColor: 'rgba(245, 158, 11, 0.6)', borderColor: '#f59e0b', borderWidth: 1 }
                ]
            },
            options: getChartOptions(0, 1.0)
        });
    } else if (viewType === 'pdf_entities') {
        const csvMetrics = latest.csv_metrics_by_type || latest.csv_recall_by_type || {};
        const pdfMetrics = latest.pdf_metrics_by_type || latest.pdf_recall_by_type || {};
        const allEntities = [...new Set([...Object.keys(csvMetrics), ...Object.keys(pdfMetrics)])];
        const sortedEntities = allEntities.sort((a, b) => (csvMetrics[b]?.f1 || csvMetrics[b]?.recall || 0) - (csvMetrics[a]?.f1 || csvMetrics[a]?.recall || 0));
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedEntities.map(formatEntityName),
                datasets: [
                    { label: 'CSV F1', data: sortedEntities.map(e => csvMetrics[e]?.f1 || 0), backgroundColor: 'rgba(99, 102, 241, 0.6)', borderColor: '#6366f1', borderWidth: 1 },
                    { label: 'PDF F1', data: sortedEntities.map(e => pdfMetrics[e]?.f1 || 0), backgroundColor: 'rgba(245, 158, 11, 0.6)', borderColor: '#f59e0b', borderWidth: 1 }
                ]
            },
            options: getChartOptions(0, 1.0)
        });
    }
}

function getChartOptions(yMin, yMax, filterPdfLines = false) {
    return {
        responsive: true, maintainAspectRatio: false, clip: false,
        layout: { padding: { top: 8, bottom: 10 } },
        elements: { line: { cubicInterpolationMode: 'monotone' } },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: false },
            htmlLegend: { containerID: 'chart-legend' },
            title: { display: false },
            tooltip: {
                enabled: false,
                mode: 'index', intersect: false,
                external: function(context) {
                    let el = document.getElementById('chart-tooltip');
                    if (!el) { el = document.createElement('div'); el.id = 'chart-tooltip'; el.className = 'chart-tooltip'; document.body.appendChild(el); }
                    const tm = context.tooltip;
                    if (tm.opacity === 0) { el.style.opacity = '0'; return; }
                    const run = currentPagedData?.[tm.dataPoints?.[0]?.dataIndex];
                    const fmt = v => v != null ? (v * 100).toFixed(1) + '%' : null;
                    const metrics = [
                        { name: 'F1',        color: '#f59e0b', csv: fmt(run?.csv_overall_f1),        pdf: fmt(run?.pdf_overall_f1) },
                        { name: 'Precision',  color: '#6366f1', csv: fmt(run?.csv_overall_precision),  pdf: fmt(run?.pdf_overall_precision) },
                        { name: 'Recall',     color: '#10b981', csv: fmt(run?.csv_overall_recall),     pdf: fmt(run?.pdf_overall_recall) }
                    ];
                    const duration = run?.duration_seconds ? formatDuration(run.duration_seconds) : '';
                    const runTime = run?.timestamp ? formatRunDateTime(run.timestamp) : '';
                    const isLLM = run?.llm_benchmark;
                    const modelName = run?.model_display_name || '';
                    let h = `<div class="chart-tooltip-title">${isLLM ? (modelName + ' ') : ''}Run ${tm.dataPoints?.[0]?.label || ''}${isLLM ? ' <span style="background:#e74c3c;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.7rem;margin-left:4px;">LLM</span>' : ''}</div>`;
                    if (duration || runTime) h += `<div class="chart-tooltip-duration">${[duration, runTime].filter(Boolean).join(' \u00B7 ')}</div>`;
                    h += `<div class="chart-tooltip-grid">`;
                    h += `<div class="chart-tooltip-header"></div><div class="chart-tooltip-header">CSV</div><div class="chart-tooltip-header">PDF</div>`;
                    const isGreen = v => v && parseFloat(v) >= 90;
                    for (const m of metrics) {
                        h += `<div class="chart-tooltip-metric"><span class="chart-tooltip-dot" style="background:${m.color}"></span>${m.name}</div>`;
                        h += `<div class="chart-tooltip-val${m.csv ? (isGreen(m.csv) ? ' green' : '') : ' muted'}">${m.csv || '--'}</div>`;
                        h += `<div class="chart-tooltip-val${m.pdf ? (isGreen(m.pdf) ? ' green' : '') : ' muted'}">${m.pdf || '--'}</div>`;
                    }
                    h += `</div>`;
                    const speed = run?.duration_seconds ? (run.samples / run.duration_seconds).toFixed(1) : null;
                    if (speed) h += `<div style="margin-top: 0.375rem; font-size: 0.75rem; color: #8b8ba7;">${speed} samples/sec</div>`;
                    el.innerHTML = h;
                    el.style.opacity = '1';
                    const rect = context.chart.canvas.getBoundingClientRect();
                    let left = rect.left + window.scrollX + tm.caretX + 12;
                    let top = rect.top + window.scrollY + tm.caretY - el.offsetHeight / 2;
                    if (left + el.offsetWidth > window.innerWidth - 8) left = rect.left + window.scrollX + tm.caretX - el.offsetWidth - 12;
                    if (top < 8) top = 8;
                    el.style.left = left + 'px';
                    el.style.top = top + 'px';
                }
            }
        },
        scales: {
            y: { min: yMin, max: yMax, grid: { color: 'rgba(139, 139, 167, 0.08)' }, ticks: { color: '#8b8ba7', callback: value => (value * 100) + '%' } },
            x: { grid: { display: false }, ticks: { color: '#8b8ba7', maxRotation: 0, minRotation: 0 } }
        }
    };
}

function switchChartTab() {
    // Both charts are always visible now - kept for compatibility
}

function renderEntityTabs(container) {
    const tabClass = tab => 'format-pill' + (currentEntityTab === tab ? ' active' : '');
    container.innerHTML = `<div class="format-filter">
        <button class="${tabClass('csv')}" onclick="switchEntityTab('csv')">CSV</button>
        <button class="${tabClass('pdf')}" onclick="switchEntityTab('pdf')">PDF</button>
        <button class="${tabClass('definitions')}" onclick="switchEntityTab('definitions')">Definitions</button>
    </div>`;
}

function switchEntityTab(tab) {
    currentEntityTab = tab;
    const chartTabs = document.getElementById('chart-tabs');
    if (chartTabs) renderEntityTabs(chartTabs);
    renderLatestTable();
}

function sortTable(column) {
    if (currentSort.column === column) currentSort.direction = currentSort.direction === 'desc' ? 'asc' : 'desc';
    else { currentSort.column = column; currentSort.direction = 'desc'; }
    renderLatestTable();
}

function renderLatestStatsGrid() {
    const latest = benchmarkData[benchmarkData.length - 1];
    const prev = getPrevRun();
    const runDate = latest.timestamp ? new Date(latest.timestamp).toLocaleString() : 'N/A';
    const duration = latest.duration_seconds ? formatDuration(latest.duration_seconds) : 'N/A';
    const samplesPerSec = latest.duration_seconds ? (latest.samples / latest.duration_seconds) : 0;
    const prevSamplesPerSec = prev && prev.duration_seconds ? (prev.samples / prev.duration_seconds) : null;

    const csvEntities = Object.keys(latest.csv_recall_by_type || {}).length;
    const totalDetections = Object.values(latest.csv_recall_by_type || {}).reduce((sum, e) => sum + (e.tp || 0), 0);
    const totalGroundTruth = Object.values(latest.csv_recall_by_type || {}).reduce((sum, e) => sum + (e.total || 0), 0);

    // Dataset size stats
    const sets = latest.sets || Math.ceil(latest.samples / 100);
    const rowsPerSet = sets > 0 ? Math.round(latest.samples / sets) : 0;
    const totalFP = Object.values(latest.csv_recall_by_type || {}).reduce((sum, e) => sum + (e.fp || 0), 0);
    const csvTotalDetections = latest.csv_total_detections || (totalDetections + totalFP);

    const csvPrecision = latest.csv_overall_precision || 0;
    const csvRecall = latest.csv_overall_recall || 0;
    const csvF1 = latest.csv_overall_f1 || 0;
    const pdfPrecision = latest.pdf_overall_precision;
    const pdfRecall = latest.pdf_overall_recall;
    const pdfF1 = latest.pdf_overall_f1;

    const prevCsvPrecision = prev ? (prev.csv_overall_precision || 0) : null;
    const prevCsvRecall = prev ? (prev.csv_overall_recall || 0) : null;
    const prevCsvF1 = prev ? (prev.csv_overall_f1 || 0) : null;
    const prevPdfPrecision = prev ? prev.pdf_overall_precision : null;
    const prevPdfRecall = prev ? prev.pdf_overall_recall : null;
    const prevPdfF1 = prev ? prev.pdf_overall_f1 : null;

    // Detect PDF-only runs
    const isPdfOnly = (latest.csv_overall_f1 == null || latest.csv_overall_f1 === 0) &&
                       (latest.csv_overall_recall == null || latest.csv_overall_recall === 0) &&
                       (pdfF1 != null || pdfRecall != null);

    const getClass = val => Math.round(val * 1000) >= 900 ? 'green' : '';
    const csvPrecisionClass = getClass(csvPrecision);
    const csvRecallClass = getClass(csvRecall);
    const csvF1Class = getClass(csvF1);
    const pdfPrecisionClass = pdfPrecision != null ? getClass(pdfPrecision) : '';
    const pdfRecallClass = pdfRecall != null ? getClass(pdfRecall) : '';
    const pdfF1Class = pdfF1 != null ? getClass(pdfF1) : '';

    const csvPrecisionDelta = prev && prevCsvPrecision ? getDelta(csvPrecision, prevCsvPrecision) : '';
    const csvRecallDelta = prev ? getDelta(csvRecall, prevCsvRecall) : '';
    const csvF1Delta = prev && prevCsvF1 ? getDelta(csvF1, prevCsvF1) : '';
    const pdfPrecisionDelta = prev && prevPdfPrecision != null && pdfPrecision != null ? getDelta(pdfPrecision, prevPdfPrecision) : '';
    const pdfRecallDelta = prev && prevPdfRecall != null && pdfRecall != null ? getDelta(pdfRecall, prevPdfRecall) : '';
    const pdfF1Delta = prev && prevPdfF1 != null && pdfF1 != null ? getDelta(pdfF1, prevPdfF1) : '';
    const speedDelta = prevSamplesPerSec ? getDelta(samplesPerSec, prevSamplesPerSec, false) : '';
    const durationDelta = prev && prev.duration_seconds ? getDurationDelta(latest.duration_seconds, prev.duration_seconds) : '';

    // Determine run type
    const hasCsv = csvF1 > 0 || csvRecall > 0 || csvPrecision > 0;
    const hasPdf = pdfF1 != null || pdfRecall != null || pdfPrecision != null;
    let runType, runTypeClass, runTypeSub;
    if (hasCsv && hasPdf) {
        runType = 'Both';
        runTypeClass = 'green';
        runTypeSub = 'CSV + PDF';
    } else if (isPdfOnly) {
        runType = 'PDF Only';
        runTypeClass = 'orange';
        runTypeSub = 'Document only';
    } else if (hasCsv) {
        runType = 'CSV Only';
        runTypeClass = 'blue';
        runTypeSub = 'Text only';
    } else {
        runType = 'Unknown';
        runTypeClass = '';
        runTypeSub = '';
    }

    // Swap display for PDF-only runs
    let precisionHtml, recallHtml, f1Html;
    if (isPdfOnly) {
        precisionHtml = `<div class="stat-label-row"><span class="stat-label">Precision</span><span class="info-icon">i<span class="info-tooltip">Of everything flagged as PII, how much was actually PII. High precision = fewer false alarms.</span></span></div><div class="stat-value ${pdfPrecisionClass}">${pdfPrecision != null ? (pdfPrecision * 100).toFixed(1) + '%' : '0.0%'}${pdfPrecisionDelta}</div><div class="stat-sub">CSV: <span class="${csvPrecisionClass}">${(csvPrecision * 100).toFixed(1)}%</span>${csvPrecisionDelta}</div>`;
        recallHtml = `<div class="stat-label-row"><span class="stat-label">Recall</span><span class="info-icon">i<span class="info-tooltip">Of all actual PII in the data, how much was found. High recall = fewer missed detections.</span></span></div><div class="stat-value ${pdfRecallClass}">${pdfRecall != null ? (pdfRecall * 100).toFixed(1) + '%' : '0.0%'}${pdfRecallDelta}</div><div class="stat-sub">CSV: <span class="${csvRecallClass}">${(csvRecall * 100).toFixed(1)}%</span>${csvRecallDelta}</div>`;
        f1Html = `<div class="stat-label-row"><span class="stat-label">F1 Score</span><span class="info-icon">i<span class="info-tooltip">The balance between precision and recall. A high F1 means both few false alarms and few missed detections.</span></span></div><div class="stat-value ${pdfF1Class}">${pdfF1 != null ? (pdfF1 * 100).toFixed(1) + '%' : '0.0%'}${pdfF1Delta}</div><div class="stat-sub">CSV: <span class="${csvF1Class}">${(csvF1 * 100).toFixed(1)}%</span>${csvF1Delta}</div>`;
    } else {
        precisionHtml = `<div class="stat-label-row"><span class="stat-label">Precision</span><span class="info-icon">i<span class="info-tooltip">Of everything flagged as PII, how much was actually PII. High precision = fewer false alarms.</span></span></div><div class="stat-value ${csvPrecisionClass}">${(csvPrecision * 100).toFixed(1)}%${csvPrecisionDelta}</div><div class="stat-sub">PDF: <span class="${pdfPrecisionClass}">${pdfPrecision != null ? (pdfPrecision * 100).toFixed(1) + '%' : '--'}</span>${pdfPrecisionDelta}</div>`;
        recallHtml = `<div class="stat-label-row"><span class="stat-label">Recall</span><span class="info-icon">i<span class="info-tooltip">Of all actual PII in the data, how much was found. High recall = fewer missed detections.</span></span></div><div class="stat-value ${csvRecallClass}">${(csvRecall * 100).toFixed(1)}%${csvRecallDelta}</div><div class="stat-sub">PDF: <span class="${pdfRecallClass}">${pdfRecall != null ? (pdfRecall * 100).toFixed(1) + '%' : '--'}</span>${pdfRecallDelta}</div>`;
        f1Html = `<div class="stat-label-row"><span class="stat-label">F1 Score</span><span class="info-icon">i<span class="info-tooltip">The balance between precision and recall. A high F1 means both few false alarms and few missed detections.</span></span></div><div class="stat-value ${csvF1Class}">${(csvF1 * 100).toFixed(1)}%${csvF1Delta}</div><div class="stat-sub">PDF: <span class="${pdfF1Class}">${pdfF1 != null ? (pdfF1 * 100).toFixed(1) + '%' : '--'}</span>${pdfF1Delta}</div>`;
    }

    document.getElementById('table-container').innerHTML = `
        <div class="stats-compact-grid">
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Run Date</div><div class="stat-value">${runDate}</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Duration</div><div class="stat-value">${duration}${durationDelta}</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Run Type</div><div class="stat-value ${runTypeClass}">${runType}</div><div class="stat-sub">${runTypeSub}</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Speed</div><div class="stat-value">${samplesPerSec.toFixed(2)}${speedDelta}</div><div class="stat-sub">samples/sec</div></div>
            <div class="stat-card" style="cursor: default;">${precisionHtml}</div>
            <div class="stat-card" style="cursor: default;">${recallHtml}</div>
            <div class="stat-card" style="cursor: default;">${f1Html}</div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Samples</div><div class="stat-value">${latest.samples.toLocaleString()}</div><div class="stat-sub">rows tested</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Sets</div><div class="stat-value">${sets}</div><div class="stat-sub">${rowsPerSet} rows/set</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Entity Types</div><div class="stat-value">${csvEntities}</div><div class="stat-sub">unique types</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Ground Truth</div><div class="stat-value">${totalGroundTruth.toLocaleString()}</div><div class="stat-sub">expected PII</div></div>
            <div class="stat-card" style="cursor: default;"><div class="stat-label">Total Detections</div><div class="stat-value">${csvTotalDetections.toLocaleString()}</div><div class="stat-sub">TP: ${totalDetections.toLocaleString()} | FP: ${totalFP.toLocaleString()}</div></div>
        </div>
    `;
}

function renderSpeedChart() {
    const speedCanvas = document.getElementById('speed-canvas');
    if (!speedCanvas) return;

    if (speedChart) speedChart.destroy();

    const speedCtx = speedCanvas.getContext('2d');
    const pagedData = getPagedData();
    currentPagedData = pagedData; // Store for tooltip access
    const filteredData = getFilteredData();
    const startIndex = filteredData.length - pagedData.length - (currentPage * PAGE_SIZE) + 1;

    speedChart = new Chart(speedCtx, {
        type: 'bar',
        data: {
            labels: pagedData.map((r, i) => `${startIndex + i}`),
            datasets: [
                { label: 'Duration (seconds)', data: pagedData.map(r => r.duration_seconds || 0), backgroundColor: 'rgba(99, 102, 241, 0.5)', borderColor: '#6366f1', borderWidth: 1, yAxisID: 'y' },
                { label: 'Samples/sec', data: pagedData.map(r => r.duration_seconds ? r.samples / r.duration_seconds : 0), type: 'line', borderColor: '#14b8a6', backgroundColor: 'transparent', tension: 0.3, pointRadius: 3, borderWidth: 1, yAxisID: 'y1' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                htmlLegend: { containerID: 'speed-chart-legend' },
                title: { display: false },
                tooltip: {
                    mode: 'index', intersect: false,
                    backgroundColor: 'rgba(37, 37, 56, 0.95)', titleColor: '#f0f0f5', bodyColor: '#8b8ba7',
                    borderColor: 'rgba(139, 139, 167, 0.15)', borderWidth: 1, padding: 12,
                    usePointStyle: true, boxPadding: 4,
                    callbacks: {
                        title: ctx => ctx[0]?.label ? `Run ${ctx[0].label}` : '',
                        label: ctx => {
                            const value = ctx.parsed.y;
                            if (ctx.dataset.label === 'Duration (seconds)') { const m = Math.floor(value / 60); const s = Math.round(value % 60); return m > 0 ? `Duration: ${m}m ${s}s` : `Duration: ${s}s`; }
                            return `Speed: ${value.toFixed(2)} samples/sec`;
                        },
                        labelPointStyle: () => ({ pointStyle: 'circle', rotation: 0 }),
                        labelColor: ctx => ({
                            backgroundColor: ctx.dataset.borderColor || ctx.dataset.backgroundColor,
                            borderColor: ctx.dataset.borderColor || ctx.dataset.backgroundColor
                        })
                    }
                }
            },
            scales: {
                y: { type: 'linear', position: 'left', grid: { color: 'rgba(139, 139, 167, 0.08)' }, ticks: { color: '#8b8ba7', callback: value => { const m = Math.floor(value / 60); const s = Math.round(value % 60); return m > 0 ? `${m}m ${s}s` : `${s}s`; } } },
                y1: { type: 'linear', position: 'right', grid: { display: false }, ticks: { color: '#8b8ba7' } },
                x: { grid: { display: false }, ticks: { color: '#8b8ba7', maxRotation: 0, minRotation: 0 } }
            }
        }
    });
}

// Entity Mapping Matrix - shows all engine entities and their dataset mappings
const ENTITY_MAPPING_MATRIX = {
    // Engine entity type -> { benchmark: mapped type, datasets: { ds1: [...fields], ds2: [...fields], ds3: [...labels] } }
    'PERSON': {
        benchmark: 'PERSON',
        datasets: {
            'Synthetic (Faker)': ['PERSON'],
            'ai4privacy (300k)': ['GIVENNAME1', 'GIVENNAME2', 'LASTNAME1', 'LASTNAME2', 'LASTNAME3', 'TITLE'],
            'Parquet (50 types)': ['first_name', 'last_name', 'user_name', 'occupation']
        },
        hasGroundTruth: true
    },
    'EMAIL_ADDRESS': {
        benchmark: 'EMAIL',
        datasets: {
            'Synthetic (Faker)': ['EMAIL'],
            'ai4privacy (300k)': ['EMAIL'],
            'Parquet (50 types)': ['email']
        },
        hasGroundTruth: true
    },
    'PHONE_NUMBER': {
        benchmark: 'PHONE',
        datasets: {
            'Synthetic (Faker)': ['PHONE'],
            'ai4privacy (300k)': ['TEL'],
            'Parquet (50 types)': ['phone_number', 'fax_number']
        },
        hasGroundTruth: true
    },
    'LOCATION': {
        benchmark: 'ADDRESS',
        datasets: {
            'Synthetic (Faker)': ['ADDRESS'],
            'ai4privacy (300k)': ['STREET', 'BUILDING', 'CITY', 'STATE', 'POSTCODE', 'COUNTRY', 'SECADDRESS'],
            'Parquet (50 types)': ['street_address', 'city', 'state', 'postcode', 'country', 'county']
        },
        hasGroundTruth: true
    },
    'URL': {
        benchmark: 'URL',
        datasets: {
            'Synthetic (Faker)': ['URL'],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['url']
        },
        hasGroundTruth: true
    },
    'SSN': {
        benchmark: 'NATIONAL_ID',
        datasets: {
            'Synthetic (Faker)': ['NATIONAL_ID'],
            'ai4privacy (300k)': ['SOCIALNUMBER'],
            'Parquet (50 types)': ['ssn']
        },
        hasGroundTruth: true,
        note: 'Deprecated: now emits NATIONAL_ID'
    },
    'NATIONAL_ID': {
        benchmark: 'NATIONAL_ID',
        datasets: {
            'Synthetic (Faker)': ['NATIONAL_ID'],
            'ai4privacy (300k)': ['SOCIALNUMBER', 'PASSPORT', 'DRIVERLICENSE'],
            'Parquet (50 types)': ['certificate_license_number', 'passport', 'passport_number', 'drivers_license', 'driver_license', 'national_id', 'tax_id', 'ssn']
        },
        hasGroundTruth: true,
        note: 'Consolidates SSN, passport, license'
    },
    'CREDIT_CARD': {
        benchmark: 'CREDIT_CARD',
        datasets: {
            'Synthetic (Faker)': ['CREDIT_CARD'],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['credit_debit_card', 'cvv']
        },
        hasGroundTruth: true
    },
    'DATE_TIME': {
        benchmark: 'DATE_TIME',
        datasets: {
            'Synthetic (Faker)': ['DATE_TIME'],
            'ai4privacy (300k)': ['DATE', 'TIME', 'BOD'],
            'Parquet (50 types)': ['date_of_birth', 'date', 'date_time', 'time']
        },
        hasGroundTruth: true
    },
    'AGE': {
        benchmark: 'AGE',
        datasets: {
            'Synthetic (Faker)': ['AGE'],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['age']
        },
        hasGroundTruth: true
    },
    'GENDER': {
        benchmark: 'GENDER',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': ['SEX'],
            'Parquet (50 types)': ['gender']
        },
        hasGroundTruth: true
    },
    'COMPANY': {
        benchmark: 'COMPANY',
        datasets: {
            'Synthetic (Faker)': ['COMPANY'],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['company_name', 'organization', 'org']
        },
        hasGroundTruth: true
    },
    'ORGANIZATION': {
        benchmark: 'COMPANY',
        datasets: {
            'Synthetic (Faker)': ['COMPANY'],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['company_name', 'organization', 'org']
        },
        hasGroundTruth: true,
        note: 'Merged with COMPANY'
    },
    'FINANCIAL': {
        benchmark: 'FINANCIAL',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['swift_bic', 'bank_routing_number', 'account_number']
        },
        hasGroundTruth: true
    },
    'IP_ADDRESS': {
        benchmark: 'IP_ADDRESS',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': ['IP'],
            'Parquet (50 types)': ['ipv4', 'ipv6']
        },
        hasGroundTruth: true
    },
    'COORDINATES': {
        benchmark: 'COORDINATES',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': ['GEOCOORD'],
            'Parquet (50 types)': ['coordinate']
        },
        hasGroundTruth: true
    },
    'MEDICAL': {
        benchmark: 'MEDICAL',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['blood_type', 'health_plan_beneficiary_number', 'medical_record_number']
        },
        hasGroundTruth: true
    },
    'NRP': {
        benchmark: 'PERSON',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': []
        },
        hasGroundTruth: false,
        note: 'Nationality/Religion/Political - mapped to PERSON'
    },
    'AWS_ACCESS_KEY': {
        benchmark: 'CREDENTIAL',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['api_key']
        },
        hasGroundTruth: true,
        note: 'Maps to CREDENTIAL benchmark'
    },
    'STRIPE_KEY': {
        benchmark: 'CREDENTIAL',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['api_key']
        },
        hasGroundTruth: true,
        note: 'Maps to CREDENTIAL benchmark'
    },
    'FACE': {
        benchmark: 'FACE',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': []
        },
        hasGroundTruth: false,
        note: 'Image-only detection'
    },
    'QR_CODE': {
        benchmark: 'QR_CODE',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': []
        },
        hasGroundTruth: false,
        note: 'Image-only detection'
    },
    'BARCODE': {
        benchmark: 'BARCODE',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': []
        },
        hasGroundTruth: false,
        note: 'Image-only detection'
    },
    'BIOMETRIC': {
        benchmark: 'BIOMETRIC',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['biometric_identifier']
        },
        hasGroundTruth: true,
        note: 'Fingerprints, facial recognition, iris'
    },
    'CREDENTIAL': {
        benchmark: 'CREDENTIAL',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': ['USERNAME', 'PASS'],
            'Parquet (50 types)': ['password', 'pin', 'api_key']
        },
        hasGroundTruth: true,
        note: 'Passwords, PINs, API keys'
    },
    'ID': {
        benchmark: 'ID',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': ['IDCARD'],
            'Parquet (50 types)': ['customer_id', 'employee_id']
        },
        hasGroundTruth: true,
        note: 'Generic identifiers'
    },
    'NETWORK': {
        benchmark: 'NETWORK',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['mac_address', 'device_identifier', 'http_cookie']
        },
        hasGroundTruth: true,
        note: 'MAC addresses, device IDs, cookies'
    },
    'VEHICLE': {
        benchmark: 'VEHICLE',
        datasets: {
            'Synthetic (Faker)': [],
            'ai4privacy (300k)': [],
            'Parquet (50 types)': ['license_plate', 'vehicle_identifier']
        },
        hasGroundTruth: true,
        note: 'VIN, license plates'
    }
};

function renderEntityMappingTable() {
    const grayBadge = 'background: rgba(139, 139, 167, 0.2); border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; color: #fff;';
    const greenBadge = 'background: rgba(124, 179, 66, 0.25); border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; color: #fff;';
    const blueBadge = 'background: rgba(99, 102, 241, 0.25); border-radius: 4px; padding: 2px 6px; font-size: 0.75rem;';
    const orangeBadge = 'background: rgba(245, 166, 35, 0.25); border-radius: 4px; padding: 2px 6px; font-size: 0.75rem;';

    let html = `
        <div style="margin-bottom: 1rem; font-size: 0.8rem; color: var(--muted);">
            <span style="${greenBadge}">✓ Has GT</span> = Ground truth available for benchmarking
            <span style="${grayBadge}; margin-left: 0.5rem;">— No</span> = No ground truth (not benchmarked)
        </div>
        <table class="entity-table" style="font-size: 0.85rem;">
            <thead>
                <tr>
                    <th style="width: 15%;">Engine Type</th>
                    <th style="width: 12%;">Benchmark Type</th>
                    <th style="width: 13%;">Synthetic (Faker)</th>
                    <th style="width: 15%;">ai4privacy (300k)</th>
                    <th style="width: 22%;">Parquet (50 types)</th>
                    <th style="width: 8%;">Ground Truth</th>
                    <th style="width: 17%;">Notes</th>
                </tr>
            </thead>
            <tbody>
    `;

    // Sort engine entities alphabetically
    const sortedEngineEntities = Object.keys(ENTITY_MAPPING_MATRIX).sort();

    for (const entityType of sortedEngineEntities) {
        const mapping = ENTITY_MAPPING_MATRIX[entityType];
        const ds1 = mapping.datasets['Synthetic (Faker)'];
        const ds2 = mapping.datasets['ai4privacy (300k)'];
        const ds3 = mapping.datasets['Parquet (50 types)'];

        const formatFields = fields => {
            if (!fields || fields.length === 0) return '<span style="color: var(--muted);">—</span>';
            return fields.map(f => `<span style="${blueBadge}">${f}</span>`).join(' ');
        };

        const gtStatus = mapping.hasGroundTruth
            ? `<span style="${greenBadge}">✓ Yes</span>`
            : `<span style="${grayBadge}">— No</span>`;

        const note = mapping.note || '';

        html += `
            <tr>
                <td><strong>${formatEntityName(entityType)}</strong></td>
                <td><span style="${orangeBadge}">${mapping.benchmark}</span></td>
                <td>${formatFields(ds1)}</td>
                <td>${formatFields(ds2)}</td>
                <td>${formatFields(ds3)}</td>
                <td>${gtStatus}</td>
                <td style="color: var(--muted); font-size: 0.8rem;">${note}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';

    return html;
}

function renderLatestTable() {
    const latest = benchmarkData[benchmarkData.length - 1];
    const prev = getPrevRun();

    // Auto-select PDF tab for PDF-only runs (when CSV metrics are null but PDF metrics exist)
    const isPdfOnly = (latest.csv_overall_f1 == null || latest.csv_overall_recall == null) &&
                       (latest.pdf_overall_f1 != null || latest.pdf_overall_recall != null);
    if (isPdfOnly && currentEntityTab === 'csv') {
        currentEntityTab = 'pdf';
    }

    let html = '';

    if (currentEntityTab === 'definitions') {
        html += renderEntityMappingTable();
        document.getElementById('table-container').innerHTML = html;
        return;
    }

    const isCsv = currentEntityTab === 'csv';
    const metricsSource = isCsv
        ? (latest.csv_metrics_by_type || latest.csv_recall_by_type)
        : (latest.pdf_metrics_by_type || latest.pdf_recall_by_type);

    if (!metricsSource || Object.keys(metricsSource).length === 0) {
        const label = isCsv ? 'CSV' : 'PDF';
        html += `<div style="text-align: center; padding: 3rem 1rem; color: var(--muted);">
            <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">No ${label} data available</div>
            <div style="font-size: 0.8rem;">Run a benchmark with ${label} format enabled to see results here.</div>
        </div>`;
        document.getElementById('table-container').innerHTML = html;
        return;
    }

    const getSortIcon = col => currentSort.column !== col ? '<span class="sort-icon">&#8645;</span>' : (currentSort.direction === 'desc' ? '<span class="sort-icon">&#8595;</span>' : '<span class="sort-icon">&#8593;</span>');
    const getSortedClass = col => currentSort.column === col ? 'sorted' : '';

    const baseStyle = 'border-radius: 4px; padding: 2px 6px; display: inline-block;';
    const greenHighlight = baseStyle + ' background: rgba(124, 179, 66, 0.25);';
    const redHighlight = baseStyle + ' background: rgba(229, 115, 115, 0.25);';
    const noHighlight = baseStyle;
    const getPrecisionStyle = v => v >= 0.9 ? greenHighlight : noHighlight;
    const getRecallStyle = v => v >= 0.9 ? greenHighlight : noHighlight;
    const getF1Style = v => v >= 0.9 ? greenHighlight : noHighlight;

    const metricsData = metricsSource;
    const prevMetrics = isCsv
        ? (prev?.csv_metrics_by_type || prev?.csv_recall_by_type || {})
        : (prev?.pdf_metrics_by_type || prev?.pdf_recall_by_type || {});

    const allEntityData = Object.entries(metricsData).map(([entity, data]) => ({
        entity,
        precision: data.precision || 0, recall: data.recall || 0, f1: data.f1 || 0,
        tp: data.tp || 0, fp: data.fp || 0, total: data.total || 0,
        prevPrecision: prevMetrics[entity]?.precision, prevRecall: prevMetrics[entity]?.recall, prevF1: prevMetrics[entity]?.f1
    }));

    const engineEntities = allEntityData.filter(e => (e.tp + e.fp) > 0);
    const gtOnlyEntities = allEntityData.filter(e => e.total > 0 && (e.tp + e.fp) === 0);

    const sortEntities = data => [...data].sort((a, b) => {
        let valA, valB;
        switch (currentSort.column) {
            case 'entity': valA = a.entity; valB = b.entity; break;
            case 'precision': valA = a.precision; valB = b.precision; break;
            case 'recall': valA = a.recall; valB = b.recall; break;
            case 'f1': valA = a.f1; valB = b.f1; break;
            case 'tp': valA = a.tp; valB = b.tp; break;
            case 'fp': valA = a.fp; valB = b.fp; break;
            case 'gt': valA = a.total; valB = b.total; break;
            default: valA = a.f1; valB = b.f1; break;
        }
        if (currentSort.column === 'entity') return currentSort.direction === 'desc' ? valB.localeCompare(valA) : valA.localeCompare(valB);
        return currentSort.direction === 'desc' ? valB - valA : valA - valB;
    });

    const renderRow = ({ entity, precision, recall, f1, tp, fp, total, prevPrecision, prevRecall, prevF1 }) => {
        const barWidth = Math.round(f1 * 100);
        const color = f1 >= 0.8 ? 'var(--green)' : f1 >= 0.6 ? 'var(--orange)' : 'var(--red)';
        return `<tr><td><strong>${formatEntityName(entity)}</strong></td><td><span style="${getPrecisionStyle(precision)}">${(precision * 100).toFixed(1)}%</span>${getDelta(precision, prevPrecision)}</td><td><span style="${getRecallStyle(recall)}">${(recall * 100).toFixed(1)}%</span>${getDelta(recall, prevRecall)}</td><td><span style="${getF1Style(f1)}">${(f1 * 100).toFixed(1)}%</span>${getDelta(f1, prevF1)}</td><td>${tp.toLocaleString()}</td><td>${fp.toLocaleString()}</td><td>${total.toLocaleString()}</td><td><div style="display: flex; align-items: center; gap: 0.5rem;"><div class="bar-bg"><div class="bar" style="width: ${barWidth}%; background: ${color};"></div></div></div></td></tr>`;
    };

    html += `<table class="entity-table" style="margin-bottom: 2rem;"><thead><tr><th class="${getSortedClass('entity')}" onclick="sortTable('entity')">Entity${getSortIcon('entity')}</th><th class="${getSortedClass('precision')}" onclick="sortTable('precision')">Precision${getSortIcon('precision')}</th><th class="${getSortedClass('recall')}" onclick="sortTable('recall')">Recall${getSortIcon('recall')}</th><th class="${getSortedClass('f1')}" onclick="sortTable('f1')">F1${getSortIcon('f1')}</th><th class="${getSortedClass('tp')}" onclick="sortTable('tp')">True Pos${getSortIcon('tp')}</th><th class="${getSortedClass('fp')}" onclick="sortTable('fp')">False Pos${getSortIcon('fp')}</th><th class="${getSortedClass('gt')}" onclick="sortTable('gt')">Expected${getSortIcon('gt')}</th><th></th></tr></thead><tbody>`;
    sortEntities(engineEntities).forEach(e => { html += renderRow(e); });
    html += '</tbody></table>';

    if (gtOnlyEntities.length > 0) {
        html += `<h2 style="margin-bottom: 0.75rem; font-size: 0.85rem; color: var(--red);">Undetected Ground Truth <span style="color: var(--muted); font-weight: 400;">(${gtOnlyEntities.length} types missed)</span></h2><table class="entity-table"><thead><tr><th class="${getSortedClass('entity')}" onclick="sortTable('entity')">Entity${getSortIcon('entity')}</th><th class="${getSortedClass('gt')}" onclick="sortTable('gt')">Ground Truth${getSortIcon('gt')}</th><th>Status</th></tr></thead><tbody>`;
        sortEntities(gtOnlyEntities).forEach(({ entity, total }) => { html += `<tr><td><strong>${formatEntityName(entity)}</strong></td><td>${total.toLocaleString()}</td><td><span style="${redHighlight}">0% detected</span></td></tr>`; });
        html += '</tbody></table>';
    }

    document.getElementById('table-container').innerHTML = html;
}

// Event listeners
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeRunDrawer(); });
window.addEventListener('DOMContentLoaded', () => { autoLoadFiles(); startBackgroundMonitoring(); initDragScroll(); });
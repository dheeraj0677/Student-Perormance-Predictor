document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const scoreContainer = document.getElementById('score-container');
    const predictionResult = document.getElementById('prediction-result');
    const resLetter = document.getElementById('res-letter');
    const resPct    = document.getElementById('res-pct');
    const resStatus = document.getElementById('res-status');
    const gaugeWrapper     = document.getElementById('gauge-wrapper');
    const gaugePlaceholder = document.getElementById('gauge-placeholder');
    const gaugeMarker      = document.getElementById('gauge-marker');
    const gaugeBubble      = document.getElementById('gauge-bubble');
    const gaugeHint        = document.getElementById('gauge-hint');
    const adviceText       = document.getElementById('advice-text');
    const placeholderContent = scoreContainer.querySelector('.placeholder-content');

    // ── Range slider live value display ─────────────────────────────────────
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const display = slider.parentElement.querySelector('.range-value');
        if (display) {
            display.textContent = slider.value;
            slider.addEventListener('input', () => { display.textContent = slider.value; });
        }
    });

    // ── Grade Converter: 0–20 → A/B/C/D/F (Neural Network output unchanged) ──
    function convertToGrade(score) {
        const pct = Math.min(100, Math.max(0, (score / 20) * 100));
        if (pct >= 90) return { letter:'A', pct, color:'#10b981', label:'Excellent',      status:'Passing' };
        if (pct >= 80) return { letter:'B', pct, color:'#84cc16', label:'Good',            status:'Passing' };
        if (pct >= 70) return { letter:'C', pct, color:'#f59e0b', label:'Average',         status:'Passing' };
        if (pct >= 60) return { letter:'D', pct, color:'#f97316', label:'Below Average',   status:'Borderline' };
        return             { letter:'F', pct, color:'#ef4444', label:'Failing',          status:'At-Risk' };
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.textContent = "Analyzing 32 Factors...";
        
        // Collect ALL form data (all 32 factors are now in the form)
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Convert numeric string fields to integers
        const numericFields = [
            'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
            'absences', 'G1', 'G2'
        ];
        numericFields.forEach(field => {
            if (data[field] !== undefined) data[field] = parseInt(data[field]);
        });

        console.log('Sending payload with all 32 factors:', data);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errBody = await response.text();
                console.error('Server error:', errBody);
                throw new Error('Prediction failed');
            }

            const result = await response.json();

            // Update UI with results
            updateDashboard(result);

        } catch (error) {
            console.error('Error:', error);
            alert('Error generating prediction. Please try again.');
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Generate Prediction & Insights";
        }
    });

    function updateDashboard(data) {
        // 1. Update Score Card with letter grade
        placeholderContent.classList.add('hidden');
        predictionResult.classList.remove('hidden');

        const g = convertToGrade(data.prediction);
        resLetter.textContent = g.letter;
        resLetter.style.color = g.color;
        resPct.textContent    = `${g.pct.toFixed(1)}%  ·  ${data.prediction.toFixed(1)} / 20  ·  ${g.label}`;
        resStatus.textContent = g.status;
        resStatus.className   = g.status === 'At-Risk'    ? 'status-indicator status-at-risk'
                              : g.status === 'Borderline' ? 'status-indicator status-borderline'
                              :                             'status-indicator status-success';

        // 2. Update Grade Gauge
        gaugePlaceholder.classList.add('hidden');
        gaugeWrapper.classList.remove('hidden');
        updateGauge(data.prediction);

        // 3. Update SHAP plot
        if (data.plot_id) {
            const shapFrame = document.getElementById('shap-frame');
            const plotContainer = document.getElementById('plot-container');
            const placeholder = plotContainer.querySelector('.placeholder-text');
            if (placeholder) placeholder.classList.add('hidden');
            shapFrame.classList.remove('hidden');
            shapFrame.src = `/api/shap-plot/${data.plot_id}`;
        }

        // 4. Update AI Improvement Report (narrative text format)
        adviceText.innerHTML = generateAIReport(data.advice, data.prediction);

        // 5. Auto-scroll to the results so user sees the prediction
        scoreContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Grade Gauge Updater ─────────────────────────────────────────────────
    function updateGauge(score) {
        const g = convertToGrade(score);
        const trackPct = g.pct; // 0–100, used directly as % along the gradient track

        setTimeout(() => { gaugeMarker.style.left = `${trackPct}%`; }, 100);

        gaugeBubble.textContent        = g.letter;
        gaugeBubble.style.background   = g.color;
        gaugeMarker.style.background   = g.color;
        gaugeMarker.style.boxShadow    = `0 0 14px ${g.color}bb`;

        const messages = {
            A: '✨ Grade A — Outstanding performance! Keep up the excellent work.',
            B: '✅ Grade B — Good performance. A little more effort to reach A!',
            C: '📊 Grade C — Average performance. Focused improvement is recommended.',
            D: '⚠️ Grade D — Borderline. Academic support is strongly advised.',
            F: '🚨 Grade F — At-Risk. Immediate intervention is required.',
        };
        gaugeHint.textContent = messages[g.letter];
        gaugeHint.style.color = g.color;
    }

    // ── AI Narrative Report Generator ────────────────────────────────────────
    function generateAIReport(advice, prediction) {
        if (!advice || advice.length === 0) {
            return `<div class="ai-response">
                <p class="ai-intro-text">✅ <strong>Excellent profile!</strong> The neural network found no significant negative contributors. Keep up the current habits to maintain strong performance.</p>
            </div>`;
        }

        const priorities = [
            { icon: '🚨', label: 'Primary Concern' },
            { icon: '⚠️', label: 'Secondary Concern' },
            { icon: '💡', label: 'Area to Watch' }
        ];

        // Estimate potential grade recovery from top negatives
        const potentialGain = advice.reduce((sum, a) => sum + Math.abs(a.impact), 0);
        const estimatedImproved = Math.min(20, prediction + potentialGain).toFixed(1);

        const g = convertToGrade(prediction);
        const estimatedPct = Math.min(100, g.pct + potentialGain * 5).toFixed(1);

        let html = `<div class="ai-response">`;

        // Intro paragraph
        html += `<p class="ai-intro-text">🤖 <strong>Neural Network Analysis:</strong> Predicted grade is
        <strong style="color:${g.color}">${g.letter} (${g.pct.toFixed(1)}%)</strong> — ${g.label}.
        The model found <strong>${advice.length} factor${advice.length > 1 ? 's' : ''}</strong> pulling the score down.
        Addressing them could push the grade toward <strong>${estimatedPct}%</strong>.</p><hr class="ai-divider">`;

        // One paragraph per factor
        advice.forEach((item, idx) => {
            const p = priorities[idx] || { icon: '📌', label: 'Additional Factor' };
            const featLabel = item.feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const impactStr = Math.abs(item.impact).toFixed(2);

            html += `<div class="ai-point">
                <div class="ai-point-header">
                    <span class="ai-priority-icon">${p.icon}</span>
                    <span class="ai-priority-label">${p.label}</span>
                    <span class="ai-feature-tag">${featLabel}</span>
                    <span class="ai-impact-badge">−${impactStr} pts</span>
                </div>
                <p class="ai-point-text">${item.advice}</p>
            </div>`;
        });

        html += `<p class="ai-footer-text">📊 <em>This report is generated by a deep neural network trained on 395 student records using SHAP explainability. Predictions update in real-time as you adjust the student profile.</em></p>`;
        html += `</div>`;

        return html;
    }
});

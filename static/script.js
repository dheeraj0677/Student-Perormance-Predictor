document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const scoreContainer = document.getElementById('score-container');
    const predictionResult = document.getElementById('prediction-result');
    const resScore = document.getElementById('res-score');
    const resStatus = document.getElementById('res-status');
    const shapFrame = document.getElementById('shap-frame');
    const adviceList = document.getElementById('advice-list');
    const placeholderContent = scoreContainer.querySelector('.placeholder-content');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.textContent = "Analyzing Factors...";
        
        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Convert numeric strings to numbers
        const numericFields = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'];
        numericFields.forEach(field => {
            if (data[field]) data[field] = parseInt(data[field]);
        });

        // Add default values for missing fields to complete the 32-factor input
        const defaults = {
            school: "GP", sex: "F", age: 18, address: "U", famsize: "GT3", Pstatus: "A",
            Medu: 4, Fedu: 4, Mjob: "other", Fjob: "other", reason: "course", guardian: "mother",
            traveltime: 1, studytime: 2, failures: 0, schoolsup: "no", famsup: "no", paid: "no",
            activities: "no", nursery: "yes", higher: "yes", internet: "yes", romantic: "no",
            famrel: 4, freetime: 3, goout: 3, Dalc: 1, Walc: 1, health: 3, absences: 4, G1: 10, G2: 10
        };

        const payload = { ...defaults, ...data };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Prediction failed');

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
        // 1. Update Score & Status
        placeholderContent.classList.add('hidden');
        predictionResult.classList.remove('hidden');
        
        resScore.textContent = data.prediction.toFixed(1);
        resStatus.textContent = data.status;
        
        if (data.status === 'At-Risk') {
            resStatus.className = 'status-indicator status-at-risk';
        } else {
            resStatus.className = 'status-indicator status-success';
        }

        // 2. Update SHAP Plot
        const plotUrl = `/api/shap-plot/${data.plot_id}`;
        shapFrame.src = plotUrl;
        shapFrame.classList.remove('hidden');
        document.querySelector('.placeholder-plot').classList.add('hidden');

        // 3. Update Advice
        adviceList.innerHTML = '';
        if (data.advice && data.advice.length > 0) {
            data.advice.forEach(item => {
                const div = document.createElement('div');
                div.className = 'advice-item';
                div.innerHTML = `
                    <h4>${item.feature.toUpperCase()}</h4>
                    <p>${item.advice}</p>
                `;
                adviceList.appendChild(div);
            });
        } else {
            adviceList.innerHTML = '<p>No critical improvement areas identified. Maintain current habits!</p>';
        }
    }
});

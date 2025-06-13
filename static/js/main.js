// Main JS for LLM Factory UI
document.addEventListener('DOMContentLoaded', function() {
  
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
  
  // Initialize tabs
  const tabEl = document.querySelector('#step-tabs');
  if (tabEl) {
    const tabs = new bootstrap.Tab(tabEl);
  }
  
  // Animate cards on scroll
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1
  });
  
  document.querySelectorAll('.step-card').forEach(card => {
    observer.observe(card);
  });
  
  // Live prediction form
  const predictionForm = document.getElementById('prediction-form');
  const resultContainer = document.getElementById('prediction-result');
  
  if (predictionForm) {
    predictionForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      const formData = new FormData(predictionForm);
      const submitBtn = predictionForm.querySelector('button[type="submit"]');
      
      // Update button state
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
      
      // Clear previous results
      resultContainer.innerHTML = '';
      
      // Send request
      fetch('/inference', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        // Reset button
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Predict Sentiment';
        
        // Positive percentage
        const positivePercentage = Math.round(data.probabilities.positive * 100);
        const negativePercentage = Math.round(data.probabilities.negative * 100);
        
        // Create result card
        const resultCard = document.createElement('div');
        resultCard.className = 'card mt-3 result-card';
        
        resultCard.innerHTML = `
          <div class="card-body">
            <h5 class="card-title">
              Prediction: 
              <span class="badge ${data.prediction === 'Positive' ? 'badge-positive' : 'badge-negative'}">
                ${data.prediction}
              </span>
              <small class="text-muted ms-2">(${(data.confidence * 100).toFixed(1)}% confidence)</small>
            </h5>
            <p class="card-text">${data.text}</p>
            
            <div class="progress progress-sentiment">
              <div class="progress-bar progress-bar-negative" role="progressbar" 
                   style="width: ${negativePercentage}%" 
                   aria-valuenow="${negativePercentage}" aria-valuemin="0" aria-valuemax="100">
                   ${negativePercentage}%
              </div>
              <div class="progress-bar progress-bar-positive" role="progressbar" 
                   style="width: ${positivePercentage}%" 
                   aria-valuenow="${positivePercentage}" aria-valuemin="0" aria-valuemax="100">
                   ${positivePercentage}%
              </div>
            </div>
            <div class="d-flex justify-content-between mt-1">
              <small>Negative</small>
              <small>Positive</small>
            </div>
          </div>
        `;
        
        resultContainer.appendChild(resultCard);
      })
      .catch(error => {
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Predict Sentiment';
        
        resultContainer.innerHTML = `
          <div class="alert alert-danger" role="alert">
            Error: ${error.message}
          </div>
        `;
      });
    });
  }
});

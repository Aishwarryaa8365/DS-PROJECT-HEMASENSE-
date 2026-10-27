/* ============================================================
   HemaSense – Prediction Page JS  |  prediction.js
   ============================================================ */

(function () {
  'use strict';

  const form       = document.getElementById('predictionForm');
  const predictBtn = document.getElementById('predictBtn');
  const btnText    = document.getElementById('btnText');
  const btnSpinner = document.getElementById('btnSpinner');
  const valAlert   = document.getElementById('validationAlert');

  form.addEventListener('submit', function (e) {
    // Run Bootstrap native validation first
    if (!form.checkValidity()) {
      e.preventDefault();
      form.classList.add('was-validated');
      valAlert.classList.remove('d-none');
      valAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    // Valid — show spinner and let the form POST naturally to Flask
    valAlert.classList.add('d-none');
    form.classList.remove('was-validated');
    btnText.classList.add('d-none');
    btnSpinner.classList.remove('d-none');
    predictBtn.disabled = true;
  });
})();

/*=====================================================
    HemaSense - Dashboard Charts
    AI-Powered Hematology Diagnostic Assistant
=====================================================*/

// Wait until the page is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  /*=========================================
        Disease Distribution Pie Chart
    =========================================*/

  const pieCanvas = document.getElementById("diseasePieChart");

  if (pieCanvas) {
    new Chart(pieCanvas, {
      type: "pie",

      data: {
        labels: [
          "Anemia",
          "Leukemia",
          "Thalassemia",
          "Iron Deficiency",
          "Normal",
        ],

        datasets: [
          {
            data: [35, 15, 12, 18, 20],

            backgroundColor: [
              "#1976D2",
              "#D32F2F",
              "#43A047",
              "#FB8C00",
              "#8E24AA",
            ],

            borderWidth: 2,
            borderColor: "#ffffff",
          },
        ],
      },

      options: {
        responsive: true,

        plugins: {
          legend: {
            position: "bottom",
          },

          title: {
            display: true,
            text: "Disease Distribution",
          },
        },
      },
    });
  }

  /*=========================================
        Monthly Predictions Line Chart
    =========================================*/

  const lineCanvas = document.getElementById("monthlyLineChart");

  if (lineCanvas) {
    new Chart(lineCanvas, {
      type: "line",

      data: {
        labels: [
          "Jan",
          "Feb",
          "Mar",
          "Apr",
          "May",
          "Jun",
          "Jul",
          "Aug",
          "Sep",
          "Oct",
          "Nov",
          "Dec",
        ],

        datasets: [
          {
            label: "Predictions",

            data: [45, 60, 72, 90, 110, 95, 130, 140, 125, 150, 170, 190],

            borderColor: "#1976D2",

            backgroundColor: "rgba(25,118,210,0.15)",

            fill: true,

            tension: 0.4,

            borderWidth: 3,

            pointBackgroundColor: "#1976D2",

            pointRadius: 5,
          },
        ],
      },

      options: {
        responsive: true,

        plugins: {
          title: {
            display: true,
            text: "Monthly Predictions",
          },
        },

        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  }

  /*=========================================
        Prediction Accuracy Bar Chart
    =========================================*/

  const barCanvas = document.getElementById("accuracyBarChart");

  if (barCanvas) {
    new Chart(barCanvas, {
      type: "bar",

      data: {
        labels: [
          "Random Forest",
          "Decision Tree",
          "SVM",
          "KNN",
          "Logistic Regression",
        ],

        datasets: [
          {
            label: "Accuracy (%)",

            data: [97, 92, 95, 91, 89],

            backgroundColor: [
              "#1976D2",
              "#43A047",
              "#FB8C00",
              "#D32F2F",
              "#8E24AA",
            ],

            borderRadius: 10,
          },
        ],
      },

      options: {
        responsive: true,

        plugins: {
          title: {
            display: true,
            text: "Model Accuracy Comparison",
          },

          legend: {
            display: false,
          },
        },

        scales: {
          y: {
            beginAtZero: true,

            max: 100,
          },
        },
      },
    });
  }
});

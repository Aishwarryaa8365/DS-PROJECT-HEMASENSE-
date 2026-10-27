// ===============================
// HemaSense Homepage JavaScript
// ===============================

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();

    const target = document.querySelector(this.getAttribute("href"));

    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
      });
    }
  });
});

// Sticky Navbar
const navbar = document.querySelector("nav");

window.addEventListener("scroll", () => {
  if (window.scrollY > 50) {
    navbar.classList.add("shadow");
  } else {
    navbar.classList.remove("shadow");
  }
});

// Active Navigation Link
const sections = document.querySelectorAll("section");
const navLinks = document.querySelectorAll(".navbar-nav .nav-link");

window.addEventListener("scroll", () => {
  let current = "";

  sections.forEach((section) => {
    const sectionTop = section.offsetTop - 120;

    if (window.pageYOffset >= sectionTop) {
      current = section.getAttribute("id");
    }
  });

  navLinks.forEach((link) => {
    link.classList.remove("active");

    if (link.getAttribute("href") === "#" + current) {
      link.classList.add("active");
    }
  });
});

// Counter Animation
const counters = document.querySelectorAll(".counter");

const startCounter = () => {
  counters.forEach((counter) => {
    const target = +counter.getAttribute("data-target");

    let count = 0;

    const speed = target / 100;

    const update = () => {
      count += speed;

      if (count < target) {
        counter.innerText = Math.ceil(count);

        requestAnimationFrame(update);
      } else {
        counter.innerText = target;
      }
    };

    update();
  });
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      startCounter();

      observer.disconnect();
    }
  });
});

const stats = document.querySelector(".statistics");

if (stats) {
  observer.observe(stats);
}

// Fade In Animation
const revealElements = document.querySelectorAll(
  ".feature-card, .disease-card, .about-section",
);

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("show");
      }
    });
  },
  {
    threshold: 0.2,
  },
);

revealElements.forEach((el) => revealObserver.observe(el));

// Back To Top Button
const topButton = document.createElement("button");

topButton.innerHTML = "↑";

topButton.id = "topBtn";

document.body.appendChild(topButton);

topButton.style.position = "fixed";
topButton.style.bottom = "20px";
topButton.style.right = "20px";
topButton.style.display = "none";
topButton.style.padding = "12px 16px";
topButton.style.border = "none";
topButton.style.borderRadius = "50%";
topButton.style.background = "#1976D2";
topButton.style.color = "#fff";
topButton.style.cursor = "pointer";
topButton.style.zIndex = "999";

window.addEventListener("scroll", () => {
  if (window.scrollY > 400) {
    topButton.style.display = "block";
  } else {
    topButton.style.display = "none";
  }
});

topButton.addEventListener("click", () => {
  window.scrollTo({
    top: 0,

    behavior: "smooth",
  });
});

console.log("HemaSense Homepage Loaded Successfully");

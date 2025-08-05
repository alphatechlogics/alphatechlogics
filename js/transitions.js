document.addEventListener("DOMContentLoaded", function () {
  const stats = document.querySelectorAll(".stat-number");

  const animateValue = (element, start, end, duration) => {
    let startTime = null;

    const step = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const value = Math.floor(progress * (end - start) + start);
      element.textContent = value + (end.toString().includes("+") ? "+" : "");
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };

    window.requestAnimationFrame(step);
  };

  const observer = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          stats.forEach((stat) => {
            const targetValue = parseInt(stat.textContent.replace(/\D/g, ""));
            animateValue(stat, 0, targetValue, 2000);
          });
          observer.disconnect();
        }
      });
    },
    { threshold: 0.5 }
  );

  const statsGrid = document.querySelector(".stats-grid");
  if (statsGrid) {
    observer.observe(statsGrid);
  }

  // Transition animations for services section

  const services = document.querySelectorAll("#services .text-box");

  services.forEach((service, index) => {
    service.style.opacity = 0;
    service.style.transition = `opacity 0.5s ease ${
      index * 0.3
    }s, transform 0.5s ease ${index * 0.3}s`;
    service.style.transform = "translateY(20px)";
  });

  const servicesObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          services.forEach((service) => {
            service.style.opacity = 1;
            service.style.transform = "translateY(0)";
          });
          observer.disconnect();
        }
      });
    },
    { threshold: 0.8 } // Increased threshold to delay animation trigger
  );

  const servicesSection = document.querySelector("#services");
  if (servicesSection) {
    servicesObserver.observe(servicesSection);
  }

  // Fade-in animations for sections

  const sections = document.querySelectorAll(
    "section, .basic-1, .basic-2, .basic-4, .stats-section, .cards-1, .accordion-1, .form-1, .project"
  );

  sections.forEach((section) => {
    section.style.opacity = 0;
    section.style.transition = "opacity 0.8s ease, transform 0.8s ease";
    section.style.transform = "translateY(20px)";
  });

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = 1;
          entry.target.style.transform = "translateY(0)";
          sectionObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  sections.forEach((section) => sectionObserver.observe(section));

  // Explicitly handle images
  // const images = document.querySelectorAll("img");

  // images.forEach((image) => {
  //   image.classList.add("animated-img");
  // });

  // Ensure basic-3 section content remains visible
  const basic3Elements = document.querySelectorAll(
    ".basic-3 .image-container, .basic-3 .text-container, .basic-3 img"
  );

  basic3Elements.forEach((element) => {
    element.style.opacity = "1";
    element.style.transform = "none";
    element.style.transition = "none";
  });
});

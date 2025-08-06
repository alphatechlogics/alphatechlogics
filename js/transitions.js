document.addEventListener("DOMContentLoaded", function () {
  const stats = document.querySelectorAll(".stat-number");

  // Check if device is mobile
  const isMobile = window.innerWidth <= 768;

  const animateValue = (element, start, end, duration, symbol = "") => {
  let startTime = null;

  const step = (timestamp) => {
    if (!startTime) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const value = Math.floor(progress * (end - start) + start);
    element.textContent = value + symbol;
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
          const end = parseInt(stat.dataset.value);
          const symbol = stat.dataset.symbol || "";
          animateValue(stat, 0, end, 2000, symbol);
        });
          observer.disconnect();
        }
      });
    },
    { threshold: isMobile ? 0.2 : 0.5 } // Lower threshold for mobile
  );

  const statsGrid = document.querySelector(".stats-grid");
  if (statsGrid) {
    observer.observe(statsGrid);
  }

  // Transition animations for services section
  const services = document.querySelectorAll("#services .text-box");
  const servicesSection = document.querySelector("#services");

  // Only set up animations if services section exists
  if (servicesSection && services.length > 0) {
    // Check if services section is already visible on load
    const rect = servicesSection.getBoundingClientRect();
    const isInitiallyVisible = rect.top < window.innerHeight && rect.bottom > 0;

    if (isInitiallyVisible) {
      // If already visible, don't hide them - just keep them visible
      services.forEach((service) => {
        service.style.opacity = 1;
        service.style.transform = "translateY(0)";
      });
    } else {
      // Only hide them if they're not currently visible
      services.forEach((service, index) => {
        service.style.opacity = 0;
        service.style.transition = `opacity 0.5s ease ${
          index * 0.2
        }s, transform 0.5s ease ${index * 0.2}s`;
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
        { 
          threshold: isMobile ? 0.05 : 0.3, // Even lower threshold for mobile
          rootMargin: isMobile ? "100px 0px" : "50px 0px" // Larger margin for earlier trigger
        }
      );

      servicesObserver.observe(servicesSection);
    }
  }

  // Fade-in animations for sections
  const sections = document.querySelectorAll(
    "section, .basic-1, .basic-2, .basic-4, .stats-section, .cards-1, .accordion-1, .form-1, .project"
  );

  sections.forEach((section) => {
    // Check if section is initially visible
    const rect = section.getBoundingClientRect();
    const isInitiallyVisible = rect.top < window.innerHeight && rect.bottom > 0;
    
    if (!isInitiallyVisible) {
      section.style.opacity = 0;
      section.style.transition = "opacity 0.8s ease, transform 0.8s ease";
      section.style.transform = "translateY(20px)";
    }
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
    { 
      threshold: isMobile ? 0.05 : 0.3, // Lower threshold for mobile
      rootMargin: isMobile ? "50px 0px" : "20px 0px" // Earlier trigger on mobile
    }
  );

  sections.forEach((section) => {
    // Only observe sections that were initially hidden
    const rect = section.getBoundingClientRect();
    const isInitiallyVisible = rect.top < window.innerHeight && rect.bottom > 0;
    if (!isInitiallyVisible) {
      sectionObserver.observe(section);
    }
  });

  // Handle window resize to recalculate mobile state
  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      // Reload observers with new mobile state if screen size changes significantly
      const newIsMobile = window.innerWidth <= 768;
      if (newIsMobile !== isMobile) {
        location.reload(); // Simple approach - reload page on significant resize
      }
    }, 250);
  });

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
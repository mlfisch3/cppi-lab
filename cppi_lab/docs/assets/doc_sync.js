/* Scroll-spy, counterpart synchronisation and hover peek (TSD section 5.2).
 *
 * In the primary viewer a QWebChannel exposes `bridge`; this script reports
 * the section at the reading line and the cppidoc:// link under the mouse.
 * In the counterpart preview there is no channel, and the host drives the
 * page through `window.cppiDoc.focusSection(id)`.
 */
(function () {
  "use strict";
  var volume = document.documentElement.getAttribute("data-volume");
  var bridge = null;
  var focused = null;
  var previewCollapsed = false;
  var ticking = false;

  function allSections() {
    return Array.prototype.slice.call(document.querySelectorAll("section[data-concept-id]"));
  }

  function markToc(id) {
    Array.prototype.forEach.call(document.querySelectorAll("#toc a"), function (link) {
      link.classList.toggle("current", link.getAttribute("href") === "#" + id);
    });
  }

  function focusSection(id) {
    var target = document.querySelector('section[data-concept-id="' + id + '"]');
    if (!target) { return false; }
    allSections().forEach(function (s) { s.classList.toggle("synced", s === target); });
    var top = target.getBoundingClientRect().top + window.scrollY - 10;
    window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
    markToc(id);
    return true;
  }

  function readingSection() {
    var line = window.innerHeight * 0.3;
    var best = null;
    allSections().forEach(function (s) {
      var rect = s.getBoundingClientRect();
      if (rect.top <= line && rect.bottom > 40) { best = s; }
    });
    var list = allSections();
    if (!best && list.length) { best = list[0]; }
    var atEnd = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
    if (atEnd && list.length) { best = list[list.length - 1]; }
    return best ? best.getAttribute("data-concept-id") : null;
  }

  function update(force) {
    ticking = false;
    var id = readingSection();
    if (id && (force || id !== focused)) {
      focused = id;
      markToc(id);
      if (bridge) { bridge.notifySectionFocused(id); }
    }
  }

  function schedule() {
    if (!ticking) { ticking = true; window.requestAnimationFrame(function () { update(false); }); }
  }

  function parseLink(anchor) {
    try {
      var url = new URL(anchor.href);
      if (url.protocol !== "cppidoc:") { return null; }
      return { volume: url.host, anchor: url.hash.replace(/^#/, "") };
    } catch (err) { return null; }
  }

  function linkFrom(event) {
    var node = event.target;
    return node && node.closest ? node.closest('a[href^="cppidoc://"]') : null;
  }

  function showCard(anchor, info) {
    var card = document.getElementById("snippet-card");
    if (!card) { return; }
    card.querySelector(".vol").textContent = info.volume;
    card.querySelector(".ttl").textContent = info.title;
    card.querySelector(".sum").textContent = info.summary;
    var rect = anchor.getBoundingClientRect();
    card.style.display = "block";
    var left = Math.min(rect.left, window.innerWidth - card.offsetWidth - 12);
    var top = rect.bottom + 8;
    if (top + card.offsetHeight > window.innerHeight - 8) { top = rect.top - card.offsetHeight - 8; }
    card.style.left = Math.max(8, left) + "px";
    card.style.top = Math.max(8, top) + "px";
  }

  function hideCard() {
    var card = document.getElementById("snippet-card");
    if (card) { card.style.display = "none"; }
  }

  window.cppiDoc = {
    volume: volume,
    focusSection: focusSection,
    currentSection: function () { return focused; },
    sectionIds: function () {
      return allSections().map(function (s) { return s.getAttribute("data-concept-id"); });
    }
  };

  function start() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        throwOnError: false
      });
    }
    document.documentElement.setAttribute("data-math", "rendered");

    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(schedule, {
        rootMargin: "0px 0px -70% 0px",
        threshold: [0, 0.01, 0.25, 0.5, 0.75, 1]
      });
      allSections().forEach(function (s) { observer.observe(s); });
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);

    document.addEventListener("mouseover", function (event) {
      var anchor = linkFrom(event);
      if (!anchor) { return; }
      var target = parseLink(anchor);
      if (!target || !bridge) { return; }
      bridge.notifyLinkHovered(target.volume, target.anchor);
      if (previewCollapsed || target.volume === volume) {
        bridge.sectionSnippet(target.volume, target.anchor, function (text) {
          showCard(anchor, JSON.parse(text));
        });
      }
    });
    document.addEventListener("mouseout", function (event) {
      if (!linkFrom(event)) { return; }
      hideCard();
      if (bridge) { bridge.notifyLinkLeft(); }
    });

    if (window.QWebChannel && window.qt && window.qt.webChannelTransport) {
      new window.QWebChannel(window.qt.webChannelTransport, function (channel) {
        bridge = channel.objects.bridge;
        previewCollapsed = bridge.previewCollapsed;
        bridge.previewCollapsedChanged.connect(function (value) { previewCollapsed = value; });
        bridge.notifyReady(volume);
        update(true);
      });
    } else {
      update(true);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();

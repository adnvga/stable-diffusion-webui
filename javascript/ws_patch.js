// ws_patch.js
(function () {
    console.log("WS fallback patch loaded");

    const OriginalWebSocket = window.WebSocket;

    window.WebSocket = function (url, protocols) {
        const ws = new OriginalWebSocket(url, protocols);

        ws.addEventListener("close", function (event) {
            if (event.code === 1006) {
                console.warn("WebSocket 1006 detected. Activating polling fallback.");
                startFallbackPolling();
            } else {
                console.log("WebSocket closed normally:", event.code);
            }
        });

        return ws;
    };

    function startFallbackPolling() {
        let attempts = 0;

        const interval = setInterval(async () => {
            try {
                const response = await fetch("/internal/progress");
                const data = await response.json();

                console.log("Polling status:", data);

                if (data.progress === 1 || data.completed === true) {
                    clearInterval(interval);
                    console.log("Generation completed via polling fallback.");
                    refreshGalleryWithoutReload();
                }

                attempts++;
                if (attempts > 600) {
                    console.warn("Polling timeout reached.");
                    clearInterval(interval);
                }

            } catch (err) {
                console.warn("Polling error:", err);
            }
        }, 1500);
    }

    function refreshGalleryWithoutReload() {
        console.log("Triggering soft UI refresh.");

        window.dispatchEvent(new Event("resize"));

        const interruptBtn = document.querySelector("button[id*='interrupt']");
        if (interruptBtn) {
            interruptBtn.click();
        }
    }

})();

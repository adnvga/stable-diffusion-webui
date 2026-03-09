console.log("🔥 [ws_safe] Frontend JS loaded.");
console.log("🚨🚨🚨 WS_SAFE VERSION 6001 (DIAGNOSTIC) 🚨🚨🚨");

(function () {

let wsFailureActive = false;
let refreshAttempted = false;
let wasEverActive = false;
let generationCounter = 0;
let currentGeneration = null;

/* ------------------------------------------------
   1️⃣ INTERCEPT submit()
------------------------------------------------ */

(function installSubmitHook(){

    function hook(){

        if (typeof window.submit !== "function") {
            console.warn("submit() not ready, retrying...");
            setTimeout(hook, 500);
            return;
        }

        const originalSubmit = window.submit;

        window.submit = function(...args){

            generationCounter++;
            currentGeneration = generationCounter;

            console.log("🚀 GENERATION START", currentGeneration);
            console.log("submit args:", args);

            const result = originalSubmit.apply(this, args);

            console.log("submit return value:", result);
            console.log("typeof:", typeof result);

            try {
                console.log("keys:", Object.keys(result || {}));
                console.log("proto:", Object.getPrototypeOf(result));
            } catch(e){}

            return result;
        };

        console.log("✅ submit hook installed");

    }

    hook();

})();

/* ------------------------------------------------
   2️⃣ INTERCEPT WEBSOCKET
------------------------------------------------ */

const OriginalWebSocket = window.WebSocket;

window.WebSocket = function (...args) {

    const ws = new OriginalWebSocket(...args);

    console.log("🔌 WS CREATED", args[0]);

    ws.addEventListener("open", () => {
        console.log("🟢 WS OPEN");
    });

    ws.addEventListener("close", (event) => {

        console.log("🔴 WS CLOSED", event.code, event.reason);

        if (event.code === 1006) {
            console.warn("⚠️ WS 1006 DETECTED");
            wsFailureActive = true;
            refreshAttempted = false;
        }
    });

    ws.addEventListener("message", function (event) {

        console.log("📡 WS RAW MESSAGE", event.data);

        try {

            const data = JSON.parse(event.data);

            console.log("📡 WS PARSED MESSAGE", data);

            if (data.msg) {
                console.log("📡 WS MSG TYPE:", data.msg);
                if (data.msg === "process_completed") {
                    wsFailureActive = false;
                }
            }

        } catch(e){
            console.log("📡 WS NON-JSON MESSAGE");
        }

    });

    return ws;
};

/* ------------------------------------------------
   3️⃣ PROGRESS POLLING
------------------------------------------------ */

const originalOpen = XMLHttpRequest.prototype.open;

XMLHttpRequest.prototype.open = function (method, url, ...rest) {

    if (url && url.includes("progress")) {

        console.log("📶 PROGRESS REQUEST", url);

        this.addEventListener("load", function () {

            try {

                const data = JSON.parse(this.responseText);

                console.log("📶 PROGRESS RESPONSE", data);

                if (data.active === true) {
                    wasEverActive = true;
                }

                if (wsFailureActive && wasEverActive && data.active === false) {

                    console.log("✅ PROGRESS DETECTED COMPLETION (WS FAILED)");
                
                    refreshAttempted = true;
                    wsFailureActive = false;
                    wasEverActive = false;
                
                    handleLostPreview();
                }

            } catch(e){
                console.log("📶 PROGRESS PARSE ERROR");
            }

        });
    }

    return originalOpen.call(this, method, url, ...rest);
};

/* ------------------------------------------------
   5️⃣ RECOVERY (placeholder)
------------------------------------------------ */

function updateTxt2ImgGallery(imageUrls) {
    const gallery = document.getElementById("txt2img_gallery");
    if (!gallery) return console.warn("Gallery not found");

    const mainImg = gallery.querySelector('[data-testid="detailed-image"]');
    const thumbs = gallery.querySelector(".thumbnails");
    const gridContainer = gallery.querySelector(".grid-container");

    if (!mainImg || !thumbs || !gridContainer) {
        console.warn("Gallery structure incomplete");
        return;
    }

    // limpiar thumbnails pequeños
    thumbs.innerHTML = "";

    // limpiar grid grande
    gridContainer.innerHTML = "";

    imageUrls.forEach((url, index) => {

        // SMALL THUMB
        const smallBtn = document.createElement("button");
        smallBtn.className = "thumbnail-item thumbnail-small svelte-1b19cri";

        const smallImg = document.createElement("img");
        smallImg.src = url;
        smallImg.className = "svelte-1b19cri";

        smallBtn.appendChild(smallImg);

        // LARGE GRID
        const largeBtn = document.createElement("button");
        largeBtn.className = "thumbnail-item thumbnail-lg svelte-1b19cri";

        const largeImg = document.createElement("img");
        largeImg.src = url;
        largeImg.className = "svelte-1b19cri";

        largeBtn.appendChild(largeImg);

        const selectImage = () => {
            mainImg.src = url;
            gallery.querySelectorAll(".thumbnail-item")
                .forEach(b => b.classList.remove("selected"));
            smallBtn.classList.add("selected");
            largeBtn.classList.add("selected");
        };

        smallBtn.onclick = selectImage;
        largeBtn.onclick = selectImage;

        if (index === 0) {
            mainImg.src = url;
            smallBtn.classList.add("selected");
            largeBtn.classList.add("selected");
        }

        thumbs.appendChild(smallBtn);
        gridContainer.appendChild(largeBtn);
    });

    console.log("Gallery updated manually:", imageUrls.length);
}
    
    
function handleLostPreview(){

    console.log("🛠 LOST PREVIEW DETECTED");

    const imgs = Array.from(document.querySelectorAll("img"));

    const outputImgs = imgs
        .map(i => i.src)
        .filter(src => src.includes("/outputs/"));

    console.log("🖼 RAW OUTPUT IMAGES", outputImgs.length);

    // 1️⃣ dedupe
    const unique = [...new Set(outputImgs)];

    // 2️⃣ separate
    const grids = unique.filter(u => u.includes("/txt2img-grids/"));
    const images = unique.filter(u => u.includes("/txt2img-images/"));

    // 3️⃣ sort images numerically
    images.sort((a, b) => {
        const getNum = u =>
            parseInt(u.match(/\/(\d+)-/)[1], 10);

        return getNum(a) - getNum(b);
    });

    // 4️⃣ final reconstruction
    const finalOrdered = [
        grids[0],
        ...images
    ];

    console.log("🧱 RECONSTRUCTED LIST", finalOrdered.length);
    console.log(finalOrdered);

    setTimeout(() => {
       updateTxt2ImgGallery(reconstructed_list);
    }, 1000);

    window.__recovered_images = finalOrdered;
}

})();
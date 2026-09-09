// Browser ka flow: theme restore karo, message API ko bhejo, response se result card/history update karo.
// HTML elements ko unke IDs se pakdo; neeche event handlers inhi references ko use karte hain.
const root = document.documentElement;
const form = document.querySelector("#analyzeForm");
const messageInput = document.querySelector("#message");
const analyzeButton = document.querySelector("#analyzeButton");
const buttonLabel = analyzeButton.querySelector(".button-label");
const characterCount = document.querySelector("#characterCount");
const historyList = document.querySelector("#historyList");
const clearButton = document.querySelector("#clearHistory");
const formError = document.querySelector("#formError");
const themeToggle = document.querySelector("#themeToggle");

// Pehle saved theme lo; nahi mili to operating system ki light/dark preference follow karo.
const storedTheme = localStorage.getItem("safeguard-theme");
const preferredTheme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
root.dataset.theme = storedTheme || preferredTheme;

// Button theme badalta hai aur localStorage mein save karta hai, taaki reload par choice yaad rahe.
themeToggle.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = nextTheme;
    localStorage.setItem("safeguard-theme", nextTheme);
});

// Input ki current JavaScript string length counter mein dikhao.
function updateCharacterCount() {
    characterCount.textContent = messageInput.value.length;
}

// Typing par counter refresh; Enter scan karta hai, Shift+Enter newline deta hai.
messageInput.addEventListener("input", updateCharacterCount);
messageInput.addEventListener("keydown", (event) => {
    // IME composition ke dauran Enter ko submit mat mano; disabled button se duplicate submit roko.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!analyzeButton.disabled) form.requestSubmit();
    }
});

// API result se DOM elements banao; alert_class card ka color aur source verdict ka origin batata hai.
function resultCard(result) {
    const article = document.createElement("article");
    article.className = `result-card result-${result.alert_class}`;

    const topline = document.createElement("div");
    topline.className = "result-topline";
    const verdict = document.createElement("span");
    verdict.className = "verdict";
    // Server/user text ko textContent se likho, taaki use executable HTML ki tarah parse na kiya jaye.
    verdict.textContent = result.status;
    const source = document.createElement("span");
    source.className = "source";
    source.textContent = result.source || "analysis";
    topline.append(verdict, source);

    const message = document.createElement("p");
    message.className = "scanned-message";
    message.textContent = result.text;

    // Verdict ke saath icon aur explanation dikhao; decorative icon screen readers se hidden hai.
    const reasonRow = document.createElement("div");
    reasonRow.className = "reason-row";
    const reasonIcon = document.createElement("span");
    reasonIcon.setAttribute("aria-hidden", "true");
    reasonIcon.textContent = result.status === "LEGIT" ? "✓" : "!";
    const reason = document.createElement("p");
    reason.textContent = result.reason;
    reasonRow.append(reasonIcon, reason);

    article.append(topline, message, reasonRow);

    // Sirf numeric confidence par percentage/bar dikhao; keyword-only result mein yeh nahi dikhti.
    if (typeof result.confidence === "number") {
        const percent = Math.round(result.confidence * 100);
        const confidence = document.createElement("div");
        confidence.className = "confidence";
        const confidenceFill = document.createElement("span");
        confidenceFill.style.width = `${percent}%`;
        confidence.append(confidenceFill);
        const caption = document.createElement("small");
        caption.textContent = `${percent}% model confidence`;
        article.append(confidence, caption);
    }

    return article;
}

// Request chalne tak button disable aur label/spinner update karo.
function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    analyzeButton.classList.toggle("is-loading", isLoading);
    buttonLabel.textContent = isLoading ? "Analyzing threat signals…" : "Scan message";
}

// Form reload roko aur async JSON request se scan result lao.
form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    setLoading(true);
    formError.hidden = true;

    try {
        // Message ko JSON body mein POST karo; await response aane tak isi async handler ko rokta hai.
        const response = await fetch("/api/v1/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({ message }),
        });
        const result = await response.json();
        // HTTP failure ko readable error mein badlo; error neeche catch block mein dikhega.
        if (!response.ok) throw new Error(result.error || "Analysis failed. Please try again.");

        // Pehle scan par empty state hatao, naya card sabse upar lagao.
        document.querySelector("#emptyState")?.remove();
        historyList.prepend(resultCard(result));
        // Browser DOM mein hard-coded maximum 10 cards rakhe hain; server ki HISTORY_LIMIT alag setting hai.
        while (historyList.children.length > 10) historyList.lastElementChild.remove();
        clearButton.hidden = false;
        messageInput.value = "";
        updateCharacterCount();
    // Network/server error ko form ke paas dikhao; finally success/failure dono mein button restore karta hai.
    } catch (error) {
        formError.textContent = error.message;
        formError.hidden = false;
    } finally {
        setLoading(false);
    }
});

// Clear par server history hatao, successful response ke baad browser list bhi khaali karo.
clearButton.addEventListener("click", async () => {
    clearButton.disabled = true;
    try {
        // Accept JSON se route ko batate hain ki redirect ke bajay JSON response chahiye.
        const response = await fetch("/clear_history", {
            method: "POST",
            headers: { "Accept": "application/json" },
        });
        if (!response.ok) throw new Error("Could not clear history.");
        historyList.replaceChildren();
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.id = "emptyState";
        // Yeh fixed developer-written HTML hai; user ka message is innerHTML mein insert nahi hota.
        empty.innerHTML = '<span class="empty-icon" aria-hidden="true">⌁</span><h3>No scans yet</h3><p>Your latest results will appear here with a clear explanation.</p>';
        historyList.append(empty);
        clearButton.hidden = true;
    } catch (error) {
        formError.textContent = error.message;
        formError.hidden = false;
    } finally {
        clearButton.disabled = false;
    }
});

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

const storedTheme = localStorage.getItem("safeguard-theme");
const preferredTheme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
root.dataset.theme = storedTheme || preferredTheme;

themeToggle.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = nextTheme;
    localStorage.setItem("safeguard-theme", nextTheme);
});

function updateCharacterCount() {
    characterCount.textContent = messageInput.value.length;
}

messageInput.addEventListener("input", updateCharacterCount);
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        form.requestSubmit();
    }
});

function resultCard(result) {
    const article = document.createElement("article");
    article.className = `result-card result-${result.alert_class}`;

    const topline = document.createElement("div");
    topline.className = "result-topline";
    const verdict = document.createElement("span");
    verdict.className = "verdict";
    verdict.textContent = result.status;
    const source = document.createElement("span");
    source.className = "source";
    source.textContent = result.source || "analysis";
    topline.append(verdict, source);

    const message = document.createElement("p");
    message.className = "scanned-message";
    message.textContent = result.text;

    const reasonRow = document.createElement("div");
    reasonRow.className = "reason-row";
    const reasonIcon = document.createElement("span");
    reasonIcon.setAttribute("aria-hidden", "true");
    reasonIcon.textContent = result.status === "LEGIT" ? "✓" : "!";
    const reason = document.createElement("p");
    reason.textContent = result.reason;
    reasonRow.append(reasonIcon, reason);

    article.append(topline, message, reasonRow);

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

function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    analyzeButton.classList.toggle("is-loading", isLoading);
    buttonLabel.textContent = isLoading ? "Analyzing threat signals…" : "Scan message";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    setLoading(true);
    formError.hidden = true;

    try {
        const response = await fetch("/api/v1/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({ message }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Analysis failed. Please try again.");

        document.querySelector("#emptyState")?.remove();
        historyList.prepend(resultCard(result));
        while (historyList.children.length > 10) historyList.lastElementChild.remove();
        clearButton.hidden = false;
        messageInput.value = "";
        updateCharacterCount();
    } catch (error) {
        formError.textContent = error.message;
        formError.hidden = false;
    } finally {
        setLoading(false);
    }
});

clearButton.addEventListener("click", async () => {
    clearButton.disabled = true;
    try {
        const response = await fetch("/clear_history", {
            method: "POST",
            headers: { "Accept": "application/json" },
        });
        if (!response.ok) throw new Error("Could not clear history.");
        historyList.replaceChildren();
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.id = "emptyState";
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

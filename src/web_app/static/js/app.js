// Browser flow: restore the theme, submit messages to the API, then update result cards and history.
// Find page elements by ID and reuse these references in the event handlers below.
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

// Prefer the saved theme; otherwise follow the operating system's light/dark preference.
const storedTheme = localStorage.getItem("safeguard-theme");
const preferredTheme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
root.dataset.theme = storedTheme || preferredTheme;

// Toggle the theme and save it in localStorage so the choice survives a reload.
themeToggle.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = nextTheme;
    localStorage.setItem("safeguard-theme", nextTheme);
});

// Display the current JavaScript string length of the input.
function updateCharacterCount() {
    characterCount.textContent = messageInput.value.length;
}

// Update the counter while typing; Enter scans, while Shift+Enter inserts a new line.
messageInput.addEventListener("input", updateCharacterCount);
messageInput.addEventListener("keydown", (event) => {
    // Do not submit during IME composition; a disabled button also prevents duplicate requests.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!analyzeButton.disabled) form.requestSubmit();
    }
});

// Build DOM elements from the API result; alert_class controls color and source identifies the verdict's origin.
function resultCard(result) {
    const article = document.createElement("article");
    article.className = `result-card result-${result.alert_class}`;

    const topline = document.createElement("div");
    topline.className = "result-topline";
    const verdict = document.createElement("span");
    verdict.className = "verdict";
    // Use textContent for server/user text so it is not interpreted as executable HTML.
    verdict.textContent = result.status;
    const source = document.createElement("span");
    source.className = "source";
    source.textContent = result.source || "analysis";
    topline.append(verdict, source);

    const message = document.createElement("p");
    message.className = "scanned-message";
    message.textContent = result.text;

    // Display an icon and explanation; hide the decorative icon from screen readers.
    const reasonRow = document.createElement("div");
    reasonRow.className = "reason-row";
    const reasonIcon = document.createElement("span");
    reasonIcon.setAttribute("aria-hidden", "true");
    reasonIcon.textContent = result.status === "LEGIT" ? "✓" : "!";
    const reason = document.createElement("p");
    reason.textContent = result.reason;
    reasonRow.append(reasonIcon, reason);

    article.append(topline, message, reasonRow);

    // Show the percentage and bar only for numeric confidence, not keyword-only results.
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

// Disable the button and update its label and spinner while the request is running.
function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    analyzeButton.classList.toggle("is-loading", isLoading);
    buttonLabel.textContent = isLoading ? "Analyzing threat signals…" : "Scan message";
}

// Prevent a page reload and retrieve the result with an asynchronous JSON request.
form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    setLoading(true);
    formError.hidden = true;

    try {
        // POST the message as JSON; await pauses this handler until the response arrives.
        const response = await fetch("/api/v1/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({ message }),
        });
        const result = await response.json();
        // Convert HTTP failures into readable errors handled by the catch block below.
        if (!response.ok) throw new Error(result.error || "Analysis failed. Please try again.");

        // Remove the empty state on the first scan and prepend the newest result card.
        document.querySelector("#emptyState")?.remove();
        historyList.prepend(resultCard(result));
        // The browser keeps at most 10 cards; the server's HISTORY_LIMIT is a separate setting.
        while (historyList.children.length > 10) historyList.lastElementChild.remove();
        clearButton.hidden = false;
        messageInput.value = "";
        updateCharacterCount();
    // Display network/server errors near the form; finally restores the button on success or failure.
    } catch (error) {
        formError.textContent = error.message;
        formError.hidden = false;
    } finally {
        setLoading(false);
    }
});

// Clear server history first, then empty the browser list after a successful response.
clearButton.addEventListener("click", async () => {
    clearButton.disabled = true;
    try {
        // Request JSON so the endpoint returns a JSON response instead of redirecting.
        const response = await fetch("/clear_history", {
            method: "POST",
            headers: { "Accept": "application/json" },
        });
        if (!response.ok) throw new Error("Could not clear history.");
        historyList.replaceChildren();
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.id = "emptyState";
        // This is fixed developer-written HTML; user messages are not inserted through innerHTML.
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

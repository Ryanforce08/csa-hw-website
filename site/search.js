let indexData = [];

const chapterSelect = document.getElementById("chapterSelect");
const exerciseSelect = document.getElementById("exerciseSelect");
const form = document.getElementById("lookup");
const listAllBtn = document.getElementById("listAll");
const resultsDiv = document.getElementById("results");

// Load index.json
async function loadIndex() {
    const res = await fetch("data/index.json");
    indexData = await res.json();
    populateChapters();
}

loadIndex();

// Populate chapter dropdown
function populateChapters() {
    const chapters = [...new Set(indexData.map(item => item.chapter))].sort((a,b)=>a-b);
    chapterSelect.innerHTML = '<option value="">Select Chapter</option>';
    chapters.forEach(ch => {
        const option = document.createElement("option");
        option.value = ch;
        option.textContent = "Chapter " + ch;
        chapterSelect.appendChild(option);
    });
    exerciseSelect.innerHTML = '<option value="">Select Exercise</option>';
}

// Populate exercises dropdown when chapter changes
chapterSelect.addEventListener("change", () => {
    const chapter = parseInt(chapterSelect.value);
    if (!chapter) {
        exerciseSelect.innerHTML = '<option value="">Select Exercise</option>';
        return;
    }
    const exercises = indexData
        .filter(item => item.chapter === chapter)
        .map(item => parseInt(item.exercise))
        .sort((a,b)=>a-b);
    
    exerciseSelect.innerHTML = '<option value="">Select Exercise</option>';
    exercises.forEach(ex => {
        const opt = document.createElement("option");
        opt.value = ex;
        opt.textContent = "Exercise " + ex;
        exerciseSelect.appendChild(opt);
    });
});

// Utility to create exercise card
function createExerciseCard(entry) {
    const card = document.createElement("div");
    card.className = "exercise-card";

    const label = document.createElement("h3");
    label.textContent = `Chapter ${entry.chapter} Exercise ${entry.exercise}`;
    card.appendChild(label);

    const img = document.createElement("img");
    img.src = entry.image;
    img.style.maxWidth = "600px";
    img.style.border = "1px solid #ccc";
    card.appendChild(img);

    const ocr = document.createElement("pre");
    ocr.textContent = entry.ocr || "(No OCR data)";
    ocr.style.whiteSpace = "pre-wrap";
    ocr.style.background = "#efefef";
    ocr.style.padding = "10px";
    card.appendChild(ocr);

    return card;
}

// Find specific exercise
form.addEventListener("submit", (e) => {
    e.preventDefault();
    const chapter = parseInt(chapterSelect.value);
    const exercise = exerciseSelect.value;

    if (!chapter || !exercise) return;

    const entry = indexData.find(item =>
        item.chapter === chapter && item.exercise === exercise
    );

    resultsDiv.innerHTML = "";

    if (!entry) {
        resultsDiv.textContent = "Exercise not found.";
        return;
    }

    resultsDiv.appendChild(createExerciseCard(entry));
});

// List all exercises in chapter
listAllBtn.addEventListener("click", () => {
    const chapter = parseInt(chapterSelect.value);
    if (!chapter) return;

    const entries = indexData
        .filter(item => item.chapter === chapter)
        .sort((a,b)=>parseInt(a.exercise)-parseInt(b.exercise));

    resultsDiv.innerHTML = "";

    if (entries.length === 0) {
        resultsDiv.textContent = "No exercises found for this chapter.";
        return;
    }

    entries.forEach(entry => {
        resultsDiv.appendChild(createExerciseCard(entry));
    });
});

let index = [];

// Load the index JSON
async function loadIndex() {
  try {
    const response = await fetch('data/index.json');
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    index = await response.json();
    console.log('Index loaded:', index);
  } catch (err) {
    console.error('Failed to load index:', err);
  }
}

// Display a single exercise
function showExercise(entry) {
  const results = document.getElementById('results');
  results.innerHTML = '';

  const img = new Image();
  img.src = entry.image;
  img.style.maxWidth = "90vw";  // scale to viewport
  img.style.maxHeight = "70vh";
  img.style.display = "block";
  img.style.margin = "auto";

  img.onerror = () => {
    results.textContent = 'Failed to load image: ' + entry.image;
  };

  results.appendChild(img);
}


// Find exercise
function findExercise(chapter, exercise) {
  return index.find(e => e.chapter === chapter && e.exercise === exercise);
}

// List all exercises in a chapter
function listExercises(chapter) {
  return index.filter(e => e.chapter === chapter);
}

// Event listeners
document.getElementById('lookup').addEventListener('submit', e => {
  e.preventDefault();
  const chapter = parseInt(document.getElementById('chapter').value);
  const exercise = document.getElementById('exercise').value;
  const entry = findExercise(chapter, exercise);
  const results = document.getElementById('results');
  if (entry) {
    showExercise(entry);
  } else {
    results.textContent = 'Exercise not found';
  }
});

document.getElementById('listAll').addEventListener('click', () => {
  const chapter = parseInt(document.getElementById('chapter').value);
  const entries = listExercises(chapter);
  const results = document.getElementById('results');
  results.innerHTML = '';
  if (entries.length === 0) {
    results.textContent = 'No exercises found for this chapter';
    return;
  }
  entries.forEach(entry => {
    const btn = document.createElement('button');
    btn.textContent = `Exercise ${entry.exercise}`;
    btn.addEventListener('click', () => showExercise(entry));
    results.appendChild(btn);
  });
});

// Load index
loadIndex();

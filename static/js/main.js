document.addEventListener('DOMContentLoaded', function() {
    let rawData = null;
    const uploadForm = document.getElementById('uploadForm');
    const configForm = document.getElementById('configForm');
    const targetColumnSelect = document.getElementById('targetColumn');
    const taskTypeSelect = document.getElementById('taskType');
    const trainButton = document.getElementById('trainButton');

    // Initially disable the config form
    configForm.querySelectorAll('select, button').forEach(element => {
        element.disabled = true;
    });

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show loading state
        const uploadButton = uploadForm.querySelector('button');
        const originalButtonText = uploadButton.innerHTML;
        uploadButton.innerHTML = 'Uploading...';
        uploadButton.disabled = true;

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Store the raw data
            rawData = data.data;

            // Enable and populate target column select
            targetColumnSelect.innerHTML = '<option value="">Select target column</option>';
            data.columns.forEach(column => {
                const option = document.createElement('option');
                option.value = column;
                option.textContent = column;
                targetColumnSelect.appendChild(option);
            });

            // Enable the config form elements
            configForm.querySelectorAll('select, button').forEach(element => {
                element.disabled = false;
            });

            // Display initial data analysis
            displayDataAnalysis({
                data: data.data,
                columns: data.columns
            });

            // Show success message
            alert('File uploaded successfully!');
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            // Reset upload button
            uploadButton.innerHTML = originalButtonText;
            uploadButton.disabled = false;
        });
    });

    // Handle target column selection
    targetColumnSelect.addEventListener('change', function() {
        if (rawData) {
            displayDataAnalysis({
                data: rawData,
                columns: Object.keys(rawData[0])
            });
        }
    });

    // Handle task type changes
    taskTypeSelect.addEventListener('change', function() {
        if (rawData) {
            displayDataAnalysis({
                data: rawData,
                columns: Object.keys(rawData[0])
            });
        }
    });

    async function trainModels(trainingData) {
        const progressBar = document.getElementById('trainingProgressBar');
        const statusText = document.getElementById('trainingStatus');
        let progress = 0;

        // Start progress animation
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 2;
                progress = Math.min(90, progress);
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${Math.round(progress)}%`;
                
                // Update status message based on progress
                if (progress < 30) {
                    statusText.textContent = 'Preprocessing data...';
                } else if (progress < 60) {
                    statusText.textContent = 'Training models...';
                } else {
                    statusText.textContent = 'Optimizing hyperparameters...';
                }
            }
        }, 1000);

        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trainingData)
            });

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            // Set progress to 100% on success
            clearInterval(progressInterval);
            progress = 100;
            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            statusText.textContent = 'Training completed successfully!';
            
            return result;
        } catch (error) {
            clearInterval(progressInterval);
            progressBar.classList.add('bg-danger');
            statusText.textContent = `Error: ${error.message}`;
            throw error;
        }
    }

    function detectFeatureTypes(data, targetColumn) {
        const features = Object.keys(data[0]).filter(col => col !== targetColumn);
        const numericalFeatures = [];
        const categoricalFeatures = [];

        features.forEach(feature => {
            // Check the first 100 values (or all if less than 100)
            const sampleSize = Math.min(100, data.length);
            const values = data.slice(0, sampleSize).map(row => row[feature]);
            
            // Check if the feature contains any non-numeric values
            const isNumeric = values.every(val => 
                val === null || 
                val === '' || 
                !isNaN(parseFloat(val))
            );

            // Check if the feature has low cardinality (few unique values)
            const uniqueValues = new Set(values.filter(val => val !== null && val !== '')).size;
            const isLowCardinality = uniqueValues <= 10;

            if (isNumeric && !isLowCardinality) {
                numericalFeatures.push(feature);
            } else {
                categoricalFeatures.push(feature);
            }
        });

        return { numericalFeatures, categoricalFeatures };
    }

    // Handle model training
    configForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!rawData) {
            alert('Please upload a dataset first');
            return;
        }

        const taskType = taskTypeSelect.value;
        const targetColumn = targetColumnSelect.value;
        
        if (!targetColumn) {
            alert('Please select a target column');
            return;
        }

        // Show training progress
        const trainingProgress = document.getElementById('trainingProgress');
        trainingProgress.style.display = 'block';
        
        // Reset progress bar state
        const progressBar = document.getElementById('trainingProgressBar');
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        progressBar.classList.remove('bg-danger');
        
        // Prepare training data
        const trainingData = {
            data: rawData,
            target_column: targetColumn,
            task_type: taskType
        };
        
        try {
            trainButton.disabled = true;
            trainButton.textContent = 'Training...';
            
            const result = await trainModels(trainingData);
            
            // Display results
            displayResults(result.results);
            
            // Show model downloads
            displayModelDownloads(result.results);
            
            updateStatus('Training completed successfully');
            alert('Training completed successfully!');
            
        } catch (error) {
            console.error('Training error:', error);
            updateStatus(`Training failed: ${error.message}`);
            alert('Error during training: ' + error.message);
        } finally {
            trainButton.disabled = false;
            trainButton.textContent = 'Start Training';
        }
    });

    function displayResults(results) {
        if (!results || Object.keys(results).length === 0) {
            console.error('No results to display');
            return;
        }

        console.log('Displaying results:', results);

        // Get the container elements
        const modelComparisonDiv = document.getElementById('modelComparison');
        const optimizationHistoryDiv = document.getElementById('optimizationHistory');

        if (!modelComparisonDiv || !optimizationHistoryDiv) {
            console.error('Results containers not found:', {
                modelComparison: !!modelComparisonDiv,
                optimizationHistory: !!optimizationHistoryDiv
            });
            return;
        }

        try {
            // Create model comparison chart
            const modelNames = Object.keys(results);
            const scores = modelNames.map(model => results[model].best_score);
            
            const modelComparisonData = [{
                x: modelNames,
                y: scores,
                type: 'bar',
                name: 'Model Performance',
                marker: {
                    color: 'rgb(55, 83, 109)'
                }
            }];
            
            const modelComparisonLayout = {
                title: 'Model Performance Comparison',
                xaxis: { 
                    title: 'Model',
                    tickangle: -45
                },
                yaxis: { 
                    title: 'Score'
                },
                margin: {
                    l: 50,
                    r: 50,
                    b: 100,
                    t: 50,
                    pad: 4
                },
                height: 400
            };

            const modelComparisonConfig = {
                responsive: true,
                displayModeBar: true
            };
            
            console.log('Creating model comparison plot...');
            Plotly.newPlot(modelComparisonDiv, modelComparisonData, modelComparisonLayout, modelComparisonConfig);
            
            // Create optimization history visualization
            const optimizationHistoryData = modelNames.map(model => {
                const history = results[model].optimization_history;
                if (!history || !history.values) {
                    console.warn(`No optimization history for ${model}`);
                    return null;
                }
                
                const trials = Array.from({ length: history.values.length }, (_, i) => i + 1);
                
                return {
                    x: trials,
                    y: history.values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: model,
                    line: {
                        width: 2
                    },
                    marker: {
                        size: 6
                    },
                    hovertemplate: `<b>${model}</b><br>` +
                                 'Trial: %{x}<br>' +
                                 'Score: %{y:.4f}<br>' +
                                 '<extra></extra>'
                };
            }).filter(trace => trace !== null);
            
            console.log('Optimization history data:', optimizationHistoryData);
            
            const optimizationHistoryLayout = {
                title: {
                    text: 'Model Optimization Progress',
                    font: {
                        size: 16
                    }
                },
                xaxis: { 
                    title: 'Trial Number',
                    tickmode: 'linear',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    zeroline: false
                },
                yaxis: { 
                    title: 'Score',
                    gridcolor: 'rgba(0,0,0,0.1)',
                    zeroline: false
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                margin: {
                    l: 60,
                    r: 20,
                    b: 60,
                    t: 40,
                    pad: 4
                },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: -0.2,
                    xanchor: 'center',
                    x: 0.5
                },
                height: 400,
                hovermode: 'closest'
            };

            const optimizationHistoryConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtons: [[
                    'zoom2d',
                    'pan2d',
                    'resetScale2d',
                    'toImage'
                ]],
                displaylogo: false
            };
            
            if (optimizationHistoryData.length > 0) {
                console.log('Creating optimization history plot...');
                Plotly.newPlot(optimizationHistoryDiv, optimizationHistoryData, optimizationHistoryLayout, optimizationHistoryConfig);
            } else {
                console.error('No valid optimization history data to plot');
                optimizationHistoryDiv.innerHTML = '<div class="alert alert-warning">No optimization history data available</div>';
            }

            // Show the downloads section
            const modelDownloads = document.getElementById('modelDownloads');
            if (modelDownloads) {
                modelDownloads.style.display = 'block';
            }

            console.log('Results displayed successfully');

        } catch (error) {
            console.error('Error displaying results:', error);
            alert('Error displaying results: ' + error.message);
        }
    }

    function displayModelDownloads(results) {
        const downloadsContainer = document.getElementById('modelDownloads');
        if (!downloadsContainer) {
            console.error('Model downloads container not found');
            return;
        }

        const downloadsList = downloadsContainer.querySelector('.model-downloads-list');
        if (!downloadsList) {
            console.error('Model downloads list not found');
            return;
        }
        
        // Show the downloads section
        downloadsContainer.style.display = 'block';
        
        // Clear previous download buttons
        downloadsList.innerHTML = '';
        
        // Add download button for each model
        Object.entries(results).forEach(([modelName, modelInfo]) => {
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
            downloadBtn.innerHTML = `
                <span>${modelName}</span>
                <span class="badge bg-primary rounded-pill">
                    Score: ${modelInfo.best_score.toFixed(4)}
                </span>
            `;
            
            downloadBtn.addEventListener('click', async () => {
                try {
                    downloadBtn.disabled = true;
                    downloadBtn.innerHTML = `<span>${modelName}</span><span>Downloading...</span>`;
                    
                    const response = await fetch(`/download_model/${modelName}`);
                    if (!response.ok) {
                        throw new Error('Failed to download model');
                    }
                    
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${modelName.toLowerCase()}_model.pkl`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    
                } catch (error) {
                    alert('Error downloading model: ' + error.message);
                } finally {
                    downloadBtn.disabled = false;
                    downloadBtn.innerHTML = `
                        <span>${modelName}</span>
                        <span class="badge bg-primary rounded-pill">
                            Score: ${modelInfo.best_score.toFixed(4)}
                        </span>
                    `;
                }
            });
            
            downloadsList.appendChild(downloadBtn);
        });
    }

    function updateStatus(message) {
        const statusMessage = document.getElementById('statusMessage');
        const lastUpdated = document.getElementById('lastUpdated');
        
        statusMessage.textContent = message;
        lastUpdated.textContent = new Date().toLocaleTimeString();
    }

    function displayDataAnalysis(data) {
        // Get the current selected values
        const targetColumn = document.getElementById('targetColumn').value;
        const taskType = document.getElementById('taskType').value;
        
        // Update basic info
        document.getElementById('targetColumnInfo').textContent = targetColumn || '-';
        document.getElementById('taskTypeInfo').textContent = taskType || '-';
        document.getElementById('datasetShape').textContent = `(${data.data.length}, ${Object.keys(data.data[0]).length})`;

        // Detect feature types
        const { numericalFeatures, categoricalFeatures } = detectFeatureTypes(data.data, targetColumn);
        
        // Update feature information
        document.getElementById('numFeatures').textContent = numericalFeatures.join(', ') || 'None';
        document.getElementById('catFeatures').textContent = categoricalFeatures.join(', ') || 'None';

        // Update DataFrame info
        const dataFrameInfo = document.getElementById('dataFrameInfo');
        let infoText = 'DataFrame Info:\n';
        if (data.columns) {
            data.columns.forEach(col => {
                const sampleValue = data.data[0][col];
                const type = typeof sampleValue;
                infoText += `${col}: ${type}\n`;
            });
        }
        dataFrameInfo.textContent = infoText;

        // Display sample data
        const sampleData = document.getElementById('sampleData');
        let tableHTML = '<table class="table table-sm table-striped">';
        
        // Headers
        tableHTML += '<thead><tr>';
        Object.keys(data.data[0]).forEach(col => {
            tableHTML += `<th>${col}</th>`;
        });
        tableHTML += '</tr></thead>';
        
        // Data rows (first 5 rows)
        tableHTML += '<tbody>';
        data.data.slice(0, 5).forEach(row => {
            tableHTML += '<tr>';
            Object.keys(data.data[0]).forEach(col => {
                tableHTML += `<td>${row[col]}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table>';
        
        sampleData.innerHTML = tableHTML;

        // Create distribution visualization based on task type
        if (targetColumn) {
            const targetValues = data.data.map(row => row[targetColumn]);
            
            if (taskType === 'classification') {
                // For classification: Create pie chart of class distribution
                const valueCounts = targetValues.reduce((acc, val) => {
                    acc[val] = (acc[val] || 0) + 1;
                    return acc;
                }, {});
                
                // Sort the values for better visualization
                const sortedEntries = Object.entries(valueCounts).sort(([a], [b]) => a - b);
                
                const chartData = [{
                    values: sortedEntries.map(([_, count]) => count),
                    labels: sortedEntries.map(([val, _]) => `Class ${val}`),
                    type: 'pie',
                    textinfo: 'label+percent',
                    hoverinfo: 'label+value+percent'
                }];
                
                const layout = {
                    title: 'Class Distribution',
                    height: 400,
                    margin: { t: 30, b: 30, l: 30, r: 30 }
                };
                
                Plotly.newPlot('classDistChart', chartData, layout);
            } else {
                // For regression: Create histogram of target values
                const chartData = [{
                    x: targetValues,
                    type: 'histogram',
                    nbinsx: 30,
                    name: 'Distribution',
                    opacity: 0.7
                }];
                
                const layout = {
                    title: 'Target Value Distribution',
                    height: 400,
                    margin: { t: 30, b: 50, l: 50, r: 30 },
                    xaxis: { title: targetColumn },
                    yaxis: { title: 'Frequency' }
                };
                
                Plotly.newPlot('classDistChart', chartData, layout);
            }
        }
    }
});

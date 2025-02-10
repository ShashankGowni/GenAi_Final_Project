async function search() {
    const query = document.getElementById('query').value;
    const category = document.getElementById('category').value;

    console.log(`Query: ${query}`);
    console.log(`Category: ${category}`);

    try {
        const response = await axios.get(`http://localhost:7071/api/Saravananfinproject?query=${query}&department=${category}`);
        console.log('Response:', response);
        const results = response.data;
        let resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = ''; // Clear previous results

        if (results && results.response) {
            const responseText = results.response;
            let documentsLinks = results.documents_links;

            // Create a section for the main response text
            let resultHtml = `<div class="response-text"><p>${responseText}</p></div>`;

            // Initialize documentsHtml as empty
            let documentsHtml = '';

            // Check if documentsLinks is valid before trying to split it
            if (documentsLinks && typeof documentsLinks === 'string' && documentsLinks.trim() !== '') {
                // Split the string based on newline characters
                const documentLinks = documentsLinks.split('\n');

                // We only want to show the actual document links, so let's process the documentLinks array
                documentLinks.forEach(link => {
                    if (link.trim()) {
                        // Remove the unwanted part: "Here are some related documents for your reference:"
                        if (link.includes("Here are some related documents for your reference:")) {
                            return; // Skip this line
                        }

                        // Split based on ' (Department: ' to get the document link and department
                        const [docLink, departmentInfo] = link.split(' (Department: ');
                        const department = departmentInfo ? departmentInfo.replace(')', '') : 'N/A';

                        // Clean up the URL (remove everything before the actual SharePoint URL)
                        const actualDocLink = docLink.includes("https://acuvatehyd.sharepoint.com") ? docLink.split("https://acuvatehyd.sharepoint.com")[1] : docLink;
                        const finalDocLink = `https://acuvatehyd.sharepoint.com${actualDocLink}`;

                        // Extract the document name from the URL
                        const docName = finalDocLink.substring(finalDocLink.lastIndexOf("/") + 1);

                        // Add the document name and download button
                        documentsHtml += `
                            <div class="document">
                                <p><strong>PDF Name:</strong> ${docName}</p>
                                <a href="${finalDocLink.trim()}" target="_blank" class="download-button">Download PDF</a>
                            </div>
                        `;
                    }
                });
            } else {
                // If documentsLinks is undefined or empty, show "No documents found."
                documentsHtml = "<p>No documents found.</p>";
            }

            // Append the response text and document links to the results section
            resultsDiv.innerHTML = resultHtml + documentsHtml;
        } else {
            resultsDiv.innerHTML = "<p>No results found.</p>";
        }
    } catch (error) {
        console.error('Error fetching data:', error);
        alert(`Error: ${error.message}`);
    }
}

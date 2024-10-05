if (typeof window.ethereum !== 'undefined') {
    const web3 = new Web3(window.ethereum);
    let contract;

    // Replace with your deployed contract address
    const contractAddress = '0x719130baa1ddf0700131980fb50f12db9fb13d3a';

    // ABI for the smart contract
    const contractABI = [
        {
            "inputs": [
                {
                    "internalType": "string",
                    "name": "_name",
                    "type": "string"
                },
                {
                    "internalType": "string",
                    "name": "_recordHash",
                    "type": "string"
                }
            ],
            "name": "addPatient",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "internalType": "uint256",
                    "name": "_id",
                    "type": "uint256"
                }
            ],
            "name": "getPatient",
            "outputs": [
                {
                    "internalType": "string",
                    "name": "",
                    "type": "string"
                },
                {
                    "internalType": "string",
                    "name": "",
                    "type": "string"
                },
                {
                    "internalType": "address",
                    "name": "",
                    "type": "address"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getTotalPatients",
            "outputs": [
                {
                    "internalType": "uint256",
                    "name": "",
                    "type": "uint256"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "patientCount",
            "outputs": [
                {
                    "internalType": "uint256",
                    "name": "",
                    "type": "uint256"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "internalType": "uint256",
                    "name": "",
                    "type": "uint256"
                }
            ],
            "name": "patients",
            "outputs": [
                {
                    "internalType": "string",
                    "name": "name",
                    "type": "string"
                },
                {
                    "internalType": "string",
                    "name": "recordHash",
                    "type": "string"
                },
                {
                    "internalType": "address",
                    "name": "owner",
                    "type": "address"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ];

    // Add these new functions to interact with updated smart contract methods

    // Get the total number of patients
    async function getTotalPatients() {
        try {
            const total = await contract.methods.getTotalPatients().call();
            console.log("Total patients:", total);
            return total;
        } catch (error) {
            console.error("Error getting total patients:", error);
        }
    }

    // Get a list of all patient IDs and names
    async function getAllPatients() {
        try {
            const total = await getTotalPatients();
            const patientList = document.getElementById('patientList');
            patientList.innerHTML = '';

            for (let i = 0; i < total; i++) {
                const patient = await contract.methods.getPatient(i).call();
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i}: ${patient[0]}`;
                patientList.appendChild(option);
            }
        } catch (error) {
            console.error("Error getting all patients:", error);
        }
    }

    // Call this function after successful MetaMask connection
    async function initializePatientList() {
        await getAllPatients();
        // Set up an event listener for the select element
        document.getElementById('patientList').addEventListener('change', (event) => {
            document.getElementById('patientID').value = event.target.value;
        });
    }

    async function init() {
        try {
            // Request MetaMask accounts access
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            contract = new web3.eth.Contract(contractABI, contractAddress);
            console.log("Connected account:", accounts[0]);

            // Set the connected account globally
            web3.eth.defaultAccount = accounts[0];

            // Enable UI elements after successful connection
            document.getElementById('submitReport').disabled = false;
            document.getElementById('fetchRecord').disabled = false;

            // Initialize the patient list
            await initializePatientList();
        } catch (error) {
            console.error("MetaMask Connection Error: ", error);
            document.getElementById('confirmationMessage').innerText = "Error connecting to MetaMask. Please try again.";
        }
    }

    // Add a patient record
    document.getElementById('submitReport').onclick = async () => {
        const patientName = document.getElementById('patientName').value;
        const symptoms = document.getElementById('symptoms').value;

        if (patientName && symptoms) {
            const recordHash = web3.utils.keccak256(symptoms);
            console.log("Record Hash:", recordHash);

            try {
                const accounts = await web3.eth.getAccounts();
                await contract.methods.addPatient(patientName, recordHash).send({ 
                    from: accounts[0]
                    // gas: 3000000 // Specify gas limit
                });
                document.getElementById('confirmationMessage').innerText = "Patient report submitted successfully!";
            } catch (error) {
                console.error("Transaction Error:", error);
                document.getElementById('confirmationMessage').innerText = "Error submitting the report.";
            }
        } else {
            document.getElementById('confirmationMessage').innerText = "Please enter patient name and symptoms.";
        }
    };

    // Fetch a patient record
    document.getElementById('fetchRecord').onclick = async () => {
        const patientID = document.getElementById('patientID').value;

        if (patientID) {
            try {
                const patient = await contract.methods.getPatient(patientID).call();
                document.getElementById('fetchedRecord').innerText = `Patient Name: ${patient[0]}, Symptoms Hash: ${patient[1]}`;
            } catch (error) {
                console.error("Fetching Error:", error);
                document.getElementById('fetchedRecord').innerText = "Error fetching the patient record.";
            }
        } else {
            document.getElementById('fetchedRecord').innerText = "Please enter a valid patient ID.";
        }
    };

    // Initialize MetaMask and contract connection
    init();

    // Add event listener for account changes
    window.ethereum.on('accountsChanged', (accounts) => {
        web3.eth.defaultAccount = accounts[0];
        console.log("Account changed to:", accounts[0]);
    });
} else {
    alert('Please install MetaMask to use this app.');
}
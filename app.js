if (typeof window.ethereum !== 'undefined') {
    const web3 = new Web3(window.ethereum);
    let contract;

    // Replace with your deployed contract address
    const contractAddress = '0x1c41476d7158d511cfd1f3bd7f6bc30c2b032a72';

    // ABI for the smart contract
    const contractABI = [
        {
            "inputs": [
                { "internalType": "string", "name": "_name", "type": "string" },
                { "internalType": "string", "name": "_recordHash", "type": "string" }
            ],
            "name": "addPatient",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{ "internalType": "uint256", "name": "_id", "type": "uint256" }],
            "name": "getPatient",
            "outputs": [
                { "internalType": "string", "name": "", "type": "string" },
                { "internalType": "string", "name": "", "type": "string" },
                { "internalType": "address", "name": "", "type": "address" }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getTotalPatients",
            "outputs": [{ "internalType": "uint256", "name": "", "type": "uint256" }],
            "stateMutability": "view",
            "type": "function"
        }
    ];

    // Initialize contract and connect to MetaMask
    async function init() {
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            contract = new web3.eth.Contract(contractABI, contractAddress);
            console.log("Connected account:", accounts[0]);
            web3.eth.defaultAccount = accounts[0];

            // Enable buttons
            document.getElementById('submitReport').disabled = false;
            document.getElementById('fetchRecord').disabled = false;

            // Initialize patient list
            await initializePatientList();
        } catch (error) {
            console.error("MetaMask Connection Error: ", error);
            alert("Error connecting to MetaMask.");
        }
    }

    // Fetch total patients and display
    async function getTotalPatients() {
        try {
            const total = await contract.methods.getTotalPatients().call();
            return total;
        } catch (error) {
            console.error("Error getting total patients:", error);
        }
    }

    // Fetch and display all patients
    async function getAllPatients() {
        try {
            const total = await getTotalPatients();
            const patientList = document.getElementById('patientList');
            patientList.innerHTML = ''; // Clear the list

            for (let i = 0; i < total; i++) {
                const patient = await contract.methods.getPatient(i).call();
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i}: ${patient[0]}`;
                patientList.appendChild(option);
            }
        } catch (error) {
            console.error("Error getting patients:", error);
        }
    }

    // Initialize the patient list on the page
    async function initializePatientList() {
        await getAllPatients();
        document.getElementById('patientList').addEventListener('change', (event) => {
            document.getElementById('patientID').value = event.target.value;
        });
    }

    // Submit new patient record
    document.getElementById('submitReport').onclick = async () => {
        const patientName = document.getElementById('patientName').value;
        const symptoms = document.getElementById('symptoms').value;
        const recordHash = web3.utils.keccak256(symptoms); // Hashing symptoms

        if (patientName && symptoms) {
            try {
                const accounts = await web3.eth.getAccounts();
                await contract.methods.addPatient(patientName, recordHash).send({ from: accounts[0] });
                alert("Patient report submitted successfully!");

                // Clear input fields
                document.getElementById('patientName').value = '';
                document.getElementById('symptoms').value = '';
                await getAllPatients(); // Refresh patient list
            } catch (error) {
                console.error("Error submitting the report:", error);
            }
        } else {
            alert("Please enter patient name and symptoms.");
        }
    };

    // Fetch patient details
    document.getElementById('fetchRecord').onclick = async () => {
        const patientID = document.getElementById('patientID').value;

        if (patientID) {
            try {
                const patient = await contract.methods.getPatient(patientID).call();
                alert(`Patient Name: ${patient[0]}\nSymptoms Hash: ${patient[1]}`);
            } catch (error) {
                console.error("Error fetching patient record:", error);
            }
        } else {
            alert("Please enter a valid patient ID.");
        }
    };

    // Initialize on load
    init();

    // Listen for MetaMask account changes
    window.ethereum.on('accountsChanged', (accounts) => {
        web3.eth.defaultAccount = accounts[0];
        console.log("Account changed to:", accounts[0]);
    });
} else {
    alert('Please install MetaMask to use this app.');
}




